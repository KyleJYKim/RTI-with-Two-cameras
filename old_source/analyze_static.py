#!/usr/bin/env python3
"""
1) Loads calib.yaml (K, dist).
2) Undistorts the static video.
3) Detects the square-with-corner-dot marker robustly.
4) Uses the dot to disambiguate orientation and orders corners.
5) Warps the inner square to a canonical size (e.g., 512x512).
6) Saves a Y-only MLIC stack (H x W x 1 x L, float16/32 in [0,1]) and per-pixel U/V means.

Output:
- mlic_y.npy: Y-only stack (H,W,1,L)
- uv_mean.npz: arrays U (H,W) and V (H,W) with per-pixel means across frames

Usage:
    python3 analyze_static.py \
        --video "./data_G/static_coin.mov" \
        --out_dir "./analysis/out_static_G" \
        --mapping "./sync/mapping_G.csv" \
        --det_scale 0.5 \
        --redetect 20 \
        --save_dtype float16 \
        --progress_every 200 \
        --size 512 \
        --every 1 \
        --detect_once \
        --debug

Notes:
- Assumes the object (coin) sits inside the inner white square.
- If detection jitters between frames, we track by homography initialized from the last good frame.

- Output MLIC is Y-only; U/V means are saved separately.
"""
import argparse, os, math, time
from pathlib import Path
import cv2 as cv
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=False)
    ap.add_argument("--out_dir", default="out_static")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--every", type=int, default=1, help="process every N-th frame")
    ap.add_argument("--detect_once", action="store_true",
                    help="Detect marker on the first good frame and reuse the same homography for all frames (fast). Use with tripod/static camera.")
    # detection tunables
    ap.add_argument("--min_area", type=float, default=1500.0, help="min contour area to accept as square")
    ap.add_argument("--max_ratio", type=float, default=1.6, help="max side-length ratio to accept as square")
    ap.add_argument("--th_block", type=int, default=31, help="adaptive threshold block size (odd)")
    ap.add_argument("--th_C", type=float, default=-5.0, help="adaptive threshold C parameter")
    ap.add_argument("--dot_min_area", type=float, default=3.0)
    ap.add_argument("--dot_max_area", type=float, default=1500.0)
    ap.add_argument("--dot_min_circ", type=float, default=0.5)
    ap.add_argument("--dot_color", type=int, default=-1, choices=[-1,0,255],
                    help="Fiducial dot polarity: 255=white, 0=black, -1=auto (try both and score)")
    ap.add_argument("--dot_max_dist_frac", type=float, default=0.22,
                    help="Max normalized distance from nearest corner (fraction of avg side)")
    ap.add_argument("--debug", action="store_true", help="dump debug masks/detections")
    ap.add_argument("--mapping", type=str, default=None, help="Optional mapping.csv to process only listed static frames")
    ap.add_argument("--det_scale", type=float, default=0.5, help="Downscale factor for detection (0.3~0.7). Corners are rescaled back.")
    ap.add_argument("--redetect", type=int, default=15, help="Run full detection every N frames; track with LK in between")
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N processed frames")
    ap.add_argument("--save_dtype", choices=["float16","float32"], default="float16",
                    help="Data type used when saving MLIC (float16 halves size, much faster)")
    # ECC alignment options
    ap.add_argument("--align_ecc", action="store_true", help="Perform additional alignment using ECC after ROI warp")
    ap.add_argument("--align_ecc_mode", choices=["euclidean", "affine", "homography"], default="affine",
                    help="ECC motion model: euclidean, affine, or homography")
    ap.add_argument("--align_ecc_ref", choices=["first", "median"], default="median",
                    help="Reference frame for ECC alignment: first or median of first 10")
    return ap.parse_args()

def _unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _intersect_lines(p1, p2, q1, q2):
    # Intersection of the (infinite) lines through p1→p2 and q1→q2
    a = p2 - p1
    b = q2 - q1
    A = np.array([[a[0], -b[0]], [a[1], -b[1]]], dtype=np.float64)
    rhs = (q1 - p1).astype(np.float64)
    try:
        t, u = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        t = 0.0
    return (p1 + t*a).astype(np.float32)

def estimate_inner_corners(gray, outer_corners, samples=25, max_offset_frac=0.35):
    """
    Given outer black-square corners (CCW), probe inward along each edge to
    find the black→white transition to the inner square. Return 4×2 inner corners.
    """
    c = outer_corners.astype(np.float32)
    ctr = c.mean(axis=0)

    # scale: average side length
    sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    side_avg = float(sides.mean())
    max_off = max(4.0, min(0.5*side_avg, max_offset_frac * side_avg))
    max_off_i = int(round(max_off))

    lines = []
    for i in range(4):
        a = c[i]
        b = c[(i+1) % 4]
        tvec = _unit(b - a)
        nvec = np.array([-tvec[1], tvec[0]], dtype=np.float32)  # +90°
        # ensure inward
        mid = 0.5*(a+b)
        if np.dot(nvec, (ctr - mid)) < 0:
            nvec = -nvec

        # sample K points along the edge and look for max +gradient along normal
        offsets = []
        for s in np.linspace(0.1, 0.9, samples):
            base = a + s*(b - a)
            prof = []
            x0, y0 = float(base[0]), float(base[1])
            for k in range(0, max_off_i+1):
                x = int(round(x0 + nvec[0]*k))
                y = int(round(y0 + nvec[1]*k))
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    prof.append(gray[y, x])
                else:
                    prof.append(0)
            if len(prof) <= 1:
                continue
            g = np.diff(np.asarray(prof, dtype=np.float32))
            kmax = int(np.argmax(g))  # strongest black→white jump
            offsets.append(kmax)

        di = float(np.median(offsets)) if offsets else max_off*0.5
        # build the shifted (inner) edge line
        p1 = a + nvec*di
        p2 = b + nvec*di
        lines.append((p1, p2))

    # intersections of adjacent shifted lines → inner corners
    inner = []
    for i in range(4):
        p1, p2 = lines[i-1]
        q1, q2 = lines[i]
        inner.append(_intersect_lines(p1, p2, q1, q2))
    return np.array(inner, dtype=np.float32)

def load_calib(yaml_path):
    fs = cv.FileStorage(yaml_path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat().ravel()
    fs.release()
    return K, dist

def synthesize_K(W, H, fov_deg=60.0):
    # pinhole guess: fx = fy = 0.5*W / tan(FOV/2), cx=W/2, cy=H/2
    fx = fy = 0.5 * W / math.tan(math.radians(fov_deg)*0.5)
    cx, cy = W*0.5, H*0.5
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist

def read_mapping_static_indices(path):
    idxs = []
    with open(path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 1:
                try:
                    idxs.append(int(parts[0]))
                except ValueError:
                    continue
    return sorted(set(idxs))

def undistort_maps(K, dist, size):
    return cv.initUndistortRectifyMap(K, dist, None, K, size, cv.CV_16SC2)

def preprocess(gray, clahe=False):
    # mild denoise + contrast normalization -> stable thresholding
    gray = cv.GaussianBlur(gray, (3,3), 0)
    if clahe:
        cla = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = cla.apply(gray)
    else:
        gray = cv.equalizeHist(gray)
    return gray

def _dot_contrast_score(gray, center, r, expect_polarity=255):
    """Return signed contrast (inner - ring). If expect_polarity==0, flip sign."""
    x, y = int(round(center[0])), int(round(center[1]))
    h, w = gray.shape
    if r < 1:
        return -1e9
    # Masks: inner disk radius r, ring r..1.8r
    R2 = int(round(1.8*r))
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - x)**2 + (Y - y)**2
    inner = dist2 <= r*r
    ring  = (dist2 > r*r) & (dist2 <= R2*R2)
    if not inner.any() or not ring.any():
        return -1e9
    cin = float(gray[inner].mean())
    cout = float(gray[ring].mean())
    score = (cin - cout)  # positive if dot brighter than surround
    if expect_polarity == 0:
        score = -score
    return score

def find_square_and_dot(bgr, *, th_block, th_C, min_area, max_ratio, dot_min_area, dot_max_area, dot_min_circ,
                        debug_dir=None, frame_idx=None, area_scale=1.0, dot_color=-1, dot_max_dist_frac=0.22):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    g = preprocess(gray, clahe=True)

    # Ensure odd block size >= 3
    th_block = max(3, th_block | 1)
    # Try 3 binarizations: adaptive (normal), adaptive (inverted), Otsu
    th_ad = cv.adaptiveThreshold(g, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, th_block, th_C)
    th_inv = 255 - th_ad
    _, th_otsu = cv.threshold(g, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    masks = []
    for m in (th_ad, th_inv, th_otsu):
        m1 = cv.morphologyEx(m, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
        m1 = cv.morphologyEx(m1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        masks.append(m1)

    contours = []
    chosen_mask = None
    for m in masks:
        cs, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cs) > len(contours):
            contours = cs; chosen_mask = m

    best = None
    best_poly = None
    for c in contours:
        area = cv.contourArea(c)
        if area < (min_area * area_scale):  # configurable with area scale
            continue
        peri = cv.arcLength(c, True)
        poly = cv.approxPolyDP(c, 0.02*peri, True)
        if len(poly) == 4 and cv.isContourConvex(poly):
            # score: area & squareness (ratio of side lengths)
            pts = poly.reshape(-1,2).astype(np.float32)
            # order arbitrary for now
            # compute side lengths
            d = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
            ratio = d.max()/max(1e-6, d.min())
            if ratio < max_ratio:  # relaxed squareness
                if best is None or area > cv.contourArea(best):
                    best = poly
                    best_poly = pts

    if best_poly is None:
        # optional debug dump
        if debug_dir is not None and frame_idx is not None:
            cv.imwrite(os.path.join(debug_dir, f"fail_mask_{frame_idx:06d}.png"), (chosen_mask if chosen_mask is not None else g))
        return None, None, chosen_mask

    # refine corners with sub-pixel accuracy
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    sub = cv.cornerSubPix(gray, best_poly.reshape(-1,1,2), (7,7), (-1,-1), crit)
    corners = sub.reshape(-1,2)

    # average side length (for normalized distance threshold)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    side_avg = float(sides.mean())
    max_dist = dot_max_dist_frac * side_avg

    # ---- robust fiducial dot detection ----
    x,y,w,h = cv.boundingRect(best_poly.astype(np.int32))
    roi = gray[y:y+h, x:x+w]
    roi_eq = cv.equalizeHist(cv.GaussianBlur(roi, (5,5), 0))

    def detect_blobs(blob_color):
        params = cv.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = float(dot_min_area * area_scale)
        params.maxArea = float(dot_max_area * area_scale)
        params.filterByCircularity = True
        params.minCircularity = float(dot_min_circ)
        params.filterByColor = True
        params.blobColor = int(blob_color)
        return cv.SimpleBlobDetector_create(params).detect(roi_eq)

    candidates = []
    colors_to_try = [dot_color] if dot_color in (0,255) else [255,0]
    for col in colors_to_try:
        for kp in detect_blobs(col):
            cx, cy = kp.pt[0] + x, kp.pt[1] + y
            # ignore blobs too far from any corner
            dmin = float(np.min(np.linalg.norm(corners - np.array([cx,cy], dtype=np.float32), axis=1)))
            if dmin > max_dist:
                continue
            r = max(1.5, kp.size / 2.0)
            sc = _dot_contrast_score(gray, (cx,cy), r, expect_polarity=col)
            # prefer high contrast and proximity to a corner
            score = sc - 0.02 * dmin
            candidates.append((score, (cx,cy)))

    dot_img_pt = None
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_sc, best_pt = candidates[0]
        # require positive contrast score
        if best_sc > 0:
            dot_img_pt = np.array(best_pt, dtype=np.float32)

    # optional debug overlay if provided
    if debug_dir is not None and frame_idx is not None:
        dbg = bgr.copy()
        if best_poly is not None:
            for p in corners.astype(int):
                cv.circle(dbg, tuple(p), 6, (0,255,0), 2)
        if dot_img_pt is not None:
            cv.circle(dbg, tuple(dot_img_pt.astype(int)), 6, (0,0,255), -1)
        cv.imwrite(os.path.join(debug_dir, f"det_{frame_idx:06d}.png"), dbg)
    return corners, dot_img_pt, chosen_mask  # return mask for debugging

def order_corners_ccw_with_dot(corners, dot_pt):
    # order corners CCW; choose index 0 as the corner with the dot
    ctr = corners.mean(axis=0)
    ang = np.arctan2(corners[:,1]-ctr[1], corners[:,0]-ctr[0])
    order = np.argsort(ang)
    ordered = corners[order]
    if dot_pt is None:
        return ordered  # fallback
    # find closest corner to dot
    di = np.argmin(np.linalg.norm(ordered - dot_pt[None,:], axis=1))
    # rotate so dot-corner is index 0
    ordered = np.roll(ordered, -di, axis=0)
    return ordered  # [dot, next, next, next] CCW

def warp_inner_square(frame, corners, size):
    # corners: OUTER square, CCW (index 0 near the fiducial).
    # We assume the detected corners are the OUTER black square corners.
    # Shrink inward by a fixed fraction to approximate the inner white square.
    # (If your marker has exact inner offsets in mm, replace this with precise offsets.)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    inner = estimate_inner_corners(gray, corners)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    H = cv.getPerspectiveTransform(inner, dst)
    view = cv.warpPerspective(frame, H, (size, size), flags=cv.INTER_LINEAR)
    return view, H, inner

def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    K, dist = (None, None)
    if args.calib:
        K, dist = load_calib(args.calib)

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit("Cannot open static video")
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if K is None:
        K, dist = synthesize_K(W, H)  # heuristic intrinsics, zero distortion
        print("[warn] No calib provided/found. Using synthetic K and zero distortion.")
    map1, map2 = undistort_maps(K, dist, (W, H))

    # Build list of frame indices to process (mapping-based or stride-based)
    frame_list = None
    if args.mapping:
        if not Path(args.mapping).exists():
            raise SystemExit(f"--mapping not found: {args.mapping}")
        frame_list = read_mapping_static_indices(args.mapping)
        print(f"[perf] Using mapping: processing {len(frame_list)} specific static frames.")

    # Choose save dtype and preallocate a write buffer (we may truncate at the end if some frames drop)
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    vis_first = None
    last_inner = None
    write_ptr = 0
    stack = None  # will allocate after indices is known

    debug_dir = None
    if args.debug:
        debug_dir = str((out/"debug").mkdir(parents=True, exist_ok=True) or (out/"debug"))

    # Tracking / re-detection control
    since_detect = 1_000_000
    last_gray = None
    fixed_H = None
    fixed_inner = None
    fixed_vis_dumped = False

    if frame_list is not None:
        indices = frame_list
    else:
        # derive indices by stride 'every'
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        indices = list(range(0, total_frames, max(1, args.every)))
    # Preallocate Y-only stack and U/V running sums
    total = len(indices)
    stack = np.empty((args.size, args.size, 1, total), dtype=save_dtype)
    u_sum = np.zeros((args.size, args.size), dtype=np.float32)
    v_sum = np.zeros((args.size, args.size), dtype=np.float32)
    uv_count = 0
    t0 = time.time()
    # ECC alignment setup
    ecc_ref_img = None
    ecc_mask = None
    if args.align_ecc:
        # Sample reference frame(s) for ECC
        ref_grays = []
        cap_ecc = cv.VideoCapture(args.video)
        if not cap_ecc.isOpened():
            raise SystemExit("Cannot open static video for ECC reference")
        for n in range(10):
            ok, frame = cap_ecc.read()
            if not ok:
                break
            und = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
            gray = cv.cvtColor(und, cv.COLOR_BGR2GRAY)
            ref_grays.append(gray)
        cap_ecc.release()
        if not ref_grays:
            raise SystemExit("No frames available for ECC reference")
        if args.align_ecc_ref == "median":
            # Compute pixelwise median of first 10 undistorted grays
            ecc_ref_img = np.median(np.stack(ref_grays, axis=0), axis=0).astype(np.uint8)
        else:  # "first"
            ecc_ref_img = ref_grays[0]
        # Build binary mask: white rectangular band, black central circle
        h, w = ecc_ref_img.shape
        mask = np.ones((h, w), np.uint8) * 255
        # Draw black circle at center, radius = min(h, w)/4
        center = (int(w/2), int(h/2))
        radius = int(min(h, w) / 4)
        cv.circle(mask, center, radius, 0, -1)
        ecc_mask = mask

    for k, idx in enumerate(indices, 1):
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        und = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
        gray_cur = cv.cvtColor(und, cv.COLOR_BGR2GRAY)

        # Fast path: reuse a single homography for all frames once found
        if args.detect_once and fixed_H is not None:
            view = cv.warpPerspective(und, fixed_H, (args.size, args.size), flags=cv.INTER_LINEAR)
            # ECC alignment if requested
            if args.align_ecc:
                try:
                    # Warp gray image for ECC
                    gray_view = cv.warpPerspective(gray_cur, fixed_H, (args.size, args.size), flags=cv.INTER_LINEAR)
                    # Resize ref/mask if needed
                    if ecc_ref_img.shape != gray_view.shape:
                        ref_ecc = cv.resize(ecc_ref_img, (args.size, args.size), interpolation=cv.INTER_LINEAR)
                        mask_ecc = cv.resize(ecc_mask, (args.size, args.size), interpolation=cv.INTER_NEAREST)
                    else:
                        ref_ecc = ecc_ref_img
                        mask_ecc = ecc_mask
                    # ECC motion model
                    if args.align_ecc_mode == "euclidean":
                        warp_mode = cv.MOTION_EUCLIDEAN
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                    elif args.align_ecc_mode == "affine":
                        warp_mode = cv.MOTION_AFFINE
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                    else:
                        warp_mode = cv.MOTION_HOMOGRAPHY
                        warp_matrix = np.eye(3, 3, dtype=np.float32)
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 1e-5)
                    cc, warp_matrix = cv.findTransformECC(ref_ecc, gray_view, warp_matrix, warp_mode, criteria, inputMask=mask_ecc)
                    if warp_mode == cv.MOTION_HOMOGRAPHY:
                        aligned = cv.warpPerspective(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                    else:
                        aligned = cv.warpAffine(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                    view = aligned
                except cv.error:
                    # ECC failed, fallback to unaligned view
                    pass
            yuv = cv.cvtColor(view, cv.COLOR_BGR2YUV)
            Y = (yuv[..., 0].astype(np.float32) / 255.0)
            U = (yuv[..., 1].astype(np.float32) / 255.0)
            V = (yuv[..., 2].astype(np.float32) / 255.0)
            stack[..., write_ptr] = Y[:, :, None].astype(save_dtype, copy=False)
            u_sum += U
            v_sum += V
            uv_count += 1
            write_ptr += 1
            if args.progress_every > 0 and (k % args.progress_every == 0 or k == total):
                elapsed = time.time() - t0; fps = k / max(1e-9, elapsed)
                print(f"[static] {k}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")
            continue

        do_detect = (since_detect >= args.redetect) or (last_inner is None)

        if do_detect:
            # Fast-path detection on downscaled image, then rescale back
            scale = float(args.det_scale)
            if scale <= 0.0 or scale >= 1.0:
                scale = 1.0
            small = und if scale == 1.0 else cv.resize(und, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            corners, dot_pt, _ = find_square_and_dot(
                small,
                th_block=args.th_block, th_C=args.th_C,
                min_area=args.min_area, max_ratio=args.max_ratio,
                dot_min_area=args.dot_min_area, dot_max_area=args.dot_max_area, dot_min_circ=args.dot_min_circ,
                debug_dir=debug_dir, frame_idx=idx, area_scale=(scale*scale),
                dot_color=args.dot_color, dot_max_dist_frac=args.dot_max_dist_frac
            )
            if corners is not None and scale != 1.0:
                corners = corners / scale
                if dot_pt is not None:
                    dot_pt = dot_pt / scale

            if corners is None:
                # could not detect; if we have last_inner try LK tracking
                if last_inner is not None and last_gray is not None:
                    p0 = last_inner.astype(np.float32).reshape(-1,1,2)
                    p1, st, err = cv.calcOpticalFlowPyrLK(last_gray, gray_cur, p0, None, winSize=(21,21))
                    if st is not None and int(st.sum()) == 4:
                        inner = p1.reshape(-1,2)
                        dst = np.array([[0,0],[args.size-1,0],[args.size-1,args.size-1],[0,args.size-1]],dtype=np.float32)
                        Ht = cv.getPerspectiveTransform(inner.astype(np.float32), dst)
                        view = cv.warpPerspective(und, Ht, (args.size, args.size))
                        yuv = cv.cvtColor(view, cv.COLOR_BGR2YUV)
                        Y = (yuv[..., 0].astype(np.float32) / 255.0)
                        U = (yuv[..., 1].astype(np.float32) / 255.0)
                        V = (yuv[..., 2].astype(np.float32) / 255.0)
                        stack[..., write_ptr] = Y[:, :, None].astype(save_dtype, copy=False)
                        u_sum += U
                        v_sum += V
                        uv_count += 1
                        write_ptr += 1
                        if args.progress_every > 0 and (k % args.progress_every == 0 or k == total):
                            elapsed = time.time() - t0
                            fps = k / max(1e-9, elapsed)
                            print(f"[static] {k}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")
                        last_gray = gray_cur
                        since_detect += 1
                        continue
                # else skip this frame
                last_gray = gray_cur
                since_detect += 1
                continue

            oc = order_corners_ccw_with_dot(corners, dot_pt)
            view, Hwarp, inner = warp_inner_square(und, oc, args.size)

            # Cache homography and inner corners if --detect_once is enabled
            if args.detect_once and fixed_H is None:
                fixed_H = Hwarp.copy()
                fixed_inner = inner.copy()
                if debug_dir is not None and not fixed_vis_dumped:
                    dbg2 = und.copy()
                    cv.polylines(dbg2, [oc.astype(int).reshape(-1,1,2)], True, (0,255,0), 1, cv.LINE_AA)
                    cv.polylines(dbg2, [inner.astype(int).reshape(-1,1,2)], True, (255,255,0), 3, cv.LINE_AA)
                    if dot_pt is not None:
                        cv.circle(dbg2, tuple(dot_pt.astype(int)), 6, (0,0,255), -1)
                    cv.imwrite(os.path.join(debug_dir, "detect_once_reference.png"), dbg2)
                    fixed_vis_dumped = True

            # --- DEBUG: overlay both outer and inner squares per-frame ---
            if debug_dir is not None:
                dbg2 = und.copy()
                # OUTER (green, thin)
                pts_outer = oc.astype(int).reshape(-1,1,2)
                cv.polylines(dbg2, [pts_outer], isClosed=True, color=(0,255,0), thickness=1, lineType=cv.LINE_AA)
                for p in oc.astype(int):
                    cv.circle(dbg2, tuple(p), 4, (0,200,0), 1, cv.LINE_AA)
                # INNER (cyan, thick)
                pts_inner = inner.astype(int).reshape(-1,1,2)
                cv.polylines(dbg2, [pts_inner], isClosed=True, color=(255,255,0), thickness=3, lineType=cv.LINE_AA)
                for p in inner.astype(int):
                    cv.circle(dbg2, tuple(p), 5, (255,255,0), -1, cv.LINE_AA)
                # Dot (red) if present
                if dot_pt is not None:
                    cv.circle(dbg2, tuple(dot_pt.astype(int)), 6, (0,0,255), -1)
                cv.imwrite(os.path.join(debug_dir, f"outer_inner_{idx:06d}.png"), dbg2)
            # ECC alignment if requested
            if args.align_ecc:
                try:
                    # Warp gray image for ECC
                    gray_view = cv.warpPerspective(gray_cur, Hwarp, (args.size, args.size), flags=cv.INTER_LINEAR)
                    if ecc_ref_img.shape != gray_view.shape:
                        ref_ecc = cv.resize(ecc_ref_img, (args.size, args.size), interpolation=cv.INTER_LINEAR)
                        mask_ecc = cv.resize(ecc_mask, (args.size, args.size), interpolation=cv.INTER_NEAREST)
                    else:
                        ref_ecc = ecc_ref_img
                        mask_ecc = ecc_mask
                    if args.align_ecc_mode == "euclidean":
                        warp_mode = cv.MOTION_EUCLIDEAN
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                    elif args.align_ecc_mode == "affine":
                        warp_mode = cv.MOTION_AFFINE
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                    else:
                        warp_mode = cv.MOTION_HOMOGRAPHY
                        warp_matrix = np.eye(3, 3, dtype=np.float32)
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 1e-5)
                    cc, warp_matrix = cv.findTransformECC(ref_ecc, gray_view, warp_matrix, warp_mode, criteria, inputMask=mask_ecc)
                    if warp_mode == cv.MOTION_HOMOGRAPHY:
                        aligned = cv.warpPerspective(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                    else:
                        aligned = cv.warpAffine(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                    view = aligned
                except cv.error:
                    # ECC failed, fallback to unaligned view
                    pass
            yuv = cv.cvtColor(view, cv.COLOR_BGR2YUV)
            Y = (yuv[..., 0].astype(np.float32) / 255.0)
            U = (yuv[..., 1].astype(np.float32) / 255.0)
            V = (yuv[..., 2].astype(np.float32) / 255.0)
            stack[..., write_ptr] = Y[:, :, None].astype(save_dtype, copy=False)
            u_sum += U
            v_sum += V
            uv_count += 1
            write_ptr += 1
            if args.progress_every > 0 and (k % args.progress_every == 0 or k == total):
                elapsed = time.time() - t0
                fps = k / max(1e-9, elapsed)
                print(f"[static] {k}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")
            last_inner = inner.copy()
            last_gray = gray_cur
            since_detect = 0

            if vis_first is None:
                vis = und.copy()
                # OUTER
                cv.polylines(vis, [oc.astype(int).reshape(-1,1,2)], True, (0,255,0), 1, cv.LINE_AA)
                for p in oc.astype(int):
                    cv.circle(vis, tuple(p), 5, (0,200,0), 1, cv.LINE_AA)
                # INNER
                cv.polylines(vis, [inner.astype(int).reshape(-1,1,2)], True, (255,255,0), 3, cv.LINE_AA)
                for p in inner.astype(int):
                    cv.circle(vis, tuple(p), 6, (255,255,0), -1, cv.LINE_AA)
                if dot_pt is not None:
                    cv.circle(vis, tuple(dot_pt.astype(int)), 6, (0,0,255), -1)
                cv.imwrite(str(out/"marker_detection_debug.png"), vis)
                vis_first = True
        else:
            # Track inner corners with LK, re-detect periodically
            p0 = last_inner.astype(np.float32).reshape(-1,1,2)
            p1, st, err = cv.calcOpticalFlowPyrLK(last_gray, gray_cur, p0, None, winSize=(21,21))
            if st is not None and int(st.sum()) == 4:
                inner = p1.reshape(-1,2)
                dst = np.array([[0,0],[args.size-1,0],[args.size-1,args.size-1],[0,args.size-1]],dtype=np.float32)
                Ht = cv.getPerspectiveTransform(inner.astype(np.float32), dst)
                view = cv.warpPerspective(und, Ht, (args.size, args.size))
                # ECC alignment if requested
                if args.align_ecc:
                    try:
                        # Warp gray image for ECC
                        gray_view = cv.warpPerspective(gray_cur, Ht, (args.size, args.size), flags=cv.INTER_LINEAR)
                        if ecc_ref_img.shape != gray_view.shape:
                            ref_ecc = cv.resize(ecc_ref_img, (args.size, args.size), interpolation=cv.INTER_LINEAR)
                            mask_ecc = cv.resize(ecc_mask, (args.size, args.size), interpolation=cv.INTER_NEAREST)
                        else:
                            ref_ecc = ecc_ref_img
                            mask_ecc = ecc_mask
                        if args.align_ecc_mode == "euclidean":
                            warp_mode = cv.MOTION_EUCLIDEAN
                            warp_matrix = np.eye(2, 3, dtype=np.float32)
                        elif args.align_ecc_mode == "affine":
                            warp_mode = cv.MOTION_AFFINE
                            warp_matrix = np.eye(2, 3, dtype=np.float32)
                        else:
                            warp_mode = cv.MOTION_HOMOGRAPHY
                            warp_matrix = np.eye(3, 3, dtype=np.float32)
                        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 1e-5)
                        cc, warp_matrix = cv.findTransformECC(ref_ecc, gray_view, warp_matrix, warp_mode, criteria, inputMask=mask_ecc)
                        if warp_mode == cv.MOTION_HOMOGRAPHY:
                            aligned = cv.warpPerspective(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                        else:
                            aligned = cv.warpAffine(view, warp_matrix, (args.size, args.size), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
                        view = aligned
                    except cv.error:
                        # ECC failed, fallback to unaligned view
                        pass
                yuv = cv.cvtColor(view, cv.COLOR_BGR2YUV)
                Y = (yuv[..., 0].astype(np.float32) / 255.0)
                U = (yuv[..., 1].astype(np.float32) / 255.0)
                V = (yuv[..., 2].astype(np.float32) / 255.0)
                stack[..., write_ptr] = Y[:, :, None].astype(save_dtype, copy=False)
                u_sum += U
                v_sum += V
                uv_count += 1
                write_ptr += 1
                if args.progress_every > 0 and (k % args.progress_every == 0 or k == total):
                    elapsed = time.time() - t0
                    fps = k / max(1e-9, elapsed)
                    print(f"[static] {k}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")
                last_inner = inner.copy()
                last_gray = gray_cur
                since_detect += 1
            else:
                # tracking failed; force detection next iteration
                since_detect = 1_000_000
                last_gray = gray_cur

    cap.release()
    if write_ptr == 0:
        raise SystemExit("No ROI frames collected.")
    # Truncate to the number of successfully written frames
    stack_out = stack[..., :write_ptr]
    # compute U/V per-pixel means
    if uv_count > 0:
        U_mean = (u_sum / float(uv_count)).astype(np.float32)
        V_mean = (v_sum / float(uv_count)).astype(np.float32)
    else:
        U_mean = np.zeros((args.size, args.size), dtype=np.float32)
        V_mean = np.zeros((args.size, args.size), dtype=np.float32)

    t_save0 = time.time()
    np.save(out/"mlic_y.npy", stack_out)
    np.savez(out/"uv_mean.npz", U=U_mean, V=V_mean)
    t_save = time.time() - t_save0
    elapsed_total = time.time() - t0
    print(f"[static] Y-MLIC saved: {out/'mlic_y.npy'} shape: {stack_out.shape} (H,W,1,L), dtype: {stack_out.dtype}")
    print(f"[static] U/V means saved: {out/'uv_mean.npz'} shape: {U_mean.shape}")
    print(f"[static] timings: save={t_save:.1f}s  total={elapsed_total:.1f}s")

if __name__ == "__main__":
    main()