#!/usr/bin/env python3
"""
1) Loads calib.yaml (K, dist).
2) Undistorts the moving-light video.
3) Detects the same square-with-dot marker; orders corners consistently.
4) Estimates homography H (marker plane -> image).
5) Decomposes H to pose (R,t) using K; builds unit light dir as -t/||t|| in marker frame.
6) Uses mapping.csv (from sync) to emit light directions aligned to the static MLIC frames.

Usage:
    python3 analyze_moving.py \
        --video "./data_G/moving_coin.mov" \
        --calib "./calib/moving.yaml" \
        --mapping "./sync/mapping.csv" \
        --out_dir "./analysis/out_moving"

Outputs:
  - out_moving/lights.npy : (L,3) array of unit directions matching static frames order.
  - debug overlays in out_moving/debug/.
"""
import argparse, csv, math, time
from pathlib import Path
import numpy as np
import cv2 as cv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=False, help="YAML with camera_matrix/dist_coeffs")
    ap.add_argument("--mapping", required=True, help="CSV from sync_by_audio.py")
    ap.add_argument("--out_dir", default="out_moving")
    ap.add_argument("--marker_size", type=float, default=70.0, help="marker outer-square side in arbitrary plane units")
    ap.add_argument("--th_block", type=int, default=31, help="adaptive threshold block size (odd)")
    ap.add_argument("--th_C", type=float, default=-5.0, help="adaptive threshold C parameter")
    ap.add_argument("--min_area", type=float, default=1500.0, help="min contour area to accept as square")
    ap.add_argument("--max_ratio", type=float, default=1.6, help="max side-length ratio to accept as square")
    ap.add_argument("--dot_min_area", type=float, default=3.0)
    ap.add_argument("--dot_max_area", type=float, default=1500.0)
    ap.add_argument("--dot_min_circ", type=float, default=0.5)
    ap.add_argument("--det_scale", type=float, default=0.6, help="Downscale factor for detection (0.3~0.9)")
    ap.add_argument("--debug", action="store_true", help="dump debug overlays")
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N processed frames")
    ap.add_argument("--emit_kept_pairs", type=str, default="kept_pairs.csv", help="Write pairs where a valid light was computed")
    ap.add_argument("--save_lights_dtype", choices=["float32","float16"], default="float16",
                    help="Data type when saving lights.npy (float16 reduces size; default float32)")
    ap.add_argument("--dot_color", type=int, default=-1, help="255=white dot, 0=black dot, -1=auto (default)")
    ap.add_argument("--dot_max_dist_frac", type=float, default=0.22, help="max fraction of marker side length from corner for dot")
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
    find the black -> white transition to the inner square. Return 4×2 inner corners.
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

def undistort_maps(K, dist, size):
    return cv.initUndistortRectifyMap(K, dist, None, K, size, cv.CV_16SC2)

def detect_marker_corners(bgr, *, th_block, th_C, min_area, max_ratio, dot_min_area, dot_max_area, dot_min_circ,
                          area_scale=1.0, dot_color=-1, dot_max_dist_frac=0.22):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    g = cv.GaussianBlur(gray, (3,3), 0)
    g = cv.equalizeHist(g)
    th_block = max(3, th_block | 1)
    th_ad = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,th_block,th_C)
    th_inv = 255 - th_ad
    _, th_otsu = cv.threshold(g,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    masks = []
    for m in (th_ad, th_inv, th_otsu):
        m1 = cv.morphologyEx(m, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)))
        m1 = cv.morphologyEx(m1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
        masks.append(m1)
    contours = []
    for m in masks:
        cs, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cs) > len(contours):
            contours = cs
    best_poly = None; area_best = 0.0
    for c in contours:
        area = cv.contourArea(c)
        if area < (min_area * area_scale): continue
        peri = cv.arcLength(c, True)
        poly = cv.approxPolyDP(c, 0.02*peri, True)
        if len(poly)==4 and cv.isContourConvex(poly):
            pts = poly.reshape(-1,2).astype(np.float32)
            d = np.linalg.norm(pts - np.roll(pts,-1,axis=0), axis=1)
            ratio = d.max()/max(1e-6,d.min())
            if ratio < max_ratio and area > area_best:
                area_best = area; best_poly = pts
    if best_poly is None:
        return None
    crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    sub = cv.cornerSubPix(gray, best_poly.reshape(-1,1,2), (7,7), (-1,-1), crit)
    corners = sub.reshape(-1,2)
    # dot
    x,y,w,h = cv.boundingRect(best_poly.astype(np.int32))
    roi = gray[y:y+h, x:x+w]

    avg_side = np.mean(np.linalg.norm(corners - np.roll(corners,-1,axis=0), axis=1))
    best_score = -1e9
    dot_pt = None
    colors_to_try = [dot_color] if dot_color in (0,255) else [0,255]
    for col in colors_to_try:
        params = cv.SimpleBlobDetector_Params()
        params.filterByArea=True
        params.minArea=float(dot_min_area*area_scale)
        params.maxArea=float(dot_max_area*area_scale)
        params.filterByCircularity=True; params.minCircularity=float(dot_min_circ)
        params.filterByColor=True; params.blobColor=col
        det = cv.SimpleBlobDetector_create(params)
        kps = det.detect(cv.equalizeHist(cv.GaussianBlur(roi,(5,5),0)))
        for kp in kps:
            cand = np.array([kp.pt[0]+x, kp.pt[1]+y], dtype=np.float32)
            dists = np.linalg.norm(corners - cand[None,:], axis=1)
            dmin = dists.min()
            if dmin > dot_max_dist_frac * avg_side:
                continue
            r = int(round(kp.size*0.5))
            cx, cy = int(cand[0])-x, int(cand[1])-y
            mask_in = np.zeros_like(roi, np.uint8)
            cv.circle(mask_in, (cx,cy), r, 255, -1)
            mask_ring = np.zeros_like(roi, np.uint8)
            cv.circle(mask_ring, (cx,cy), int(r*1.8), 255, -1)
            mask_ring = mask_ring - mask_in
            mean_in = cv.mean(roi, mask=mask_in)[0]
            mean_ring = cv.mean(roi, mask=mask_ring)[0]
            contrast = mean_in - mean_ring
            if col==0: contrast = -contrast
            score = contrast - 0.2*dmin
            if score > best_score:
                best_score = score
                dot_pt = cand

    ctr = corners.mean(axis=0)
    ang = np.arctan2(corners[:,1]-ctr[1], corners[:,0]-ctr[0])
    order = np.argsort(ang)
    oc = corners[order]
    if dot_pt is not None:
        di = np.argmin(np.linalg.norm(oc - dot_pt[None,:], axis=1))
        oc = np.roll(oc, -di, axis=0)

    inner = estimate_inner_corners(gray, oc.astype(np.float32))
    return inner.astype(np.float32), oc.astype(np.float32), dot_pt

def _v_ij(H, i, j):
    # Helper to build Zhang constraints: v_ij row from homography H
    h = H.T  # so h[i] is i-th column of original H
    return np.array([
        h[i,0]*h[j,0], h[i,0]*h[j,1]+h[i,1]*h[j,0], h[i,1]*h[j,1],
        h[i,2]*h[j,0]+h[i,0]*h[j,2], h[i,2]*h[j,1]+h[i,1]*h[j,2], h[i,2]*h[j,2]
    ], dtype=np.float64)

def estimate_K_from_homographies(H_list):
    """
    Zhang-style intrinsics estimation from multiple homographies.
    Solve Vb=0 for b=[B11,B12,B22,B13,B23,B33], then recover K from B.
    """
    V = []
    for H in H_list:
        v12 = _v_ij(H, 0, 1)
        v11 = _v_ij(H, 0, 0)
        v22 = _v_ij(H, 1, 1)
        V.append(v12)
        V.append(v11 - v22)
    V = np.vstack(V)
    # SVD smallest singular vector
    _, _, vt = np.linalg.svd(V)
    b = vt[-1, :]
    B11, B12, B22, B13, B23, B33 = b
    # Recover intrinsics from B = K^{-T} K^{-1}
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    fx = math.sqrt(lam / B11)
    fy = math.sqrt(lam * B11 / (B11*B22 - B12**2))
    s  = -B12 * fx**2 * fy / lam
    cx = s*v0/fy - B13*fx**2/lam
    cy = v0
    K = np.array([[fx, s,  cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def pose_from_homography(K, H):
    # H = K [r1 r2 t], up to scale. Recover r1,r2,t then make R orthonormal.
    Kinv = np.linalg.inv(K)
    h1,h2,h3 = H[:,0], H[:,1], H[:,2]
    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    t  = lam * (Kinv @ h3)
    r3 = np.cross(r1, r2)
    R  = np.column_stack([r1, r2, r3])
    # Orthonormalize via SVD (closest rotation)
    U,S,Vt = np.linalg.svd(R)
    R = U @ Vt
    # ensure right-handed
    if np.linalg.det(R) < 0:
        R[:,2] *= -1
    return R, t

def main():
    args = parse_args()
    out = Path(args.out_dir); (out/"debug").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    det_scale = float(args.det_scale)
    if det_scale <= 0.0 or det_scale >= 1.0:
        det_scale = 1.0
    K, dist = (None, None)
    if args.calib:
        K, dist = load_calib(args.calib)

    # canonical marker plane coordinates (CCW, with index 0 = dot corner)
    s = args.marker_size
    X = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)  # Z=0 plane

    # read mapping (static_idx, static_ts, moving_idx, moving_ts)
    pairs = []
    with open(args.mapping,"r") as f:
        next(f)
        for line in f:
            si, sts, mi, mts = line.strip().split(",")
            pairs.append((int(si), float(sts), int(mi), float(mts)))
    # keep unique moving indices in order of appearance for speed
    needed_m = [p[2] for p in pairs]
    unique_m, firstpos = np.unique(needed_m, return_index=True)
    unique_m = unique_m[np.argsort(firstpos)]

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit("Cannot open moving video")
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # If no K, estimate it from several marker homographies first (no undistortion).
    if K is None:
        print("[warn] No calib provided/found. Estimating K from marker across frames (Zhang).")
        # Sample up to 25 distinct moving frames where we can detect the marker
        # Use the same plane canonical X as below
        s = args.marker_size
        X = np.array([[0,0],[s,0],[s,s],[0,s]], dtype=np.float32)
        H_list = []
        n_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        # spread ~40 candidates across timeline
        picks = np.unique(np.round(np.linspace(0, max(0, n_total-1), 40)).astype(int))
        for j in picks:
            cap.set(cv.CAP_PROP_POS_FRAMES, j)
            ok, frm = cap.read()
            if not ok: continue
            small_ak = frm if det_scale == 1.0 else cv.resize(frm, None, fx=det_scale, fy=det_scale, interpolation=cv.INTER_AREA)
            inner_c, outer_c, _ = detect_marker_corners(
                small_ak,
                th_block=args.th_block, th_C=args.th_C,
                min_area=args.min_area, max_ratio=args.max_ratio,
                dot_min_area=args.dot_min_area, dot_max_area=args.dot_max_area, dot_min_circ=args.dot_min_circ,
                area_scale=(det_scale*det_scale), dot_color=args.dot_color, dot_max_dist_frac=args.dot_max_dist_frac
            )
            if inner_c is not None and det_scale != 1.0:
                inner_c = inner_c / det_scale
            # (optional) minimal progress each 10 picks
            if len(H_list) % 2 == 0 and len(H_list) > 0:
                elapsed = time.time() - t0
                print(f"[autoK] homographies={len(H_list)}  elapsed={elapsed:.1f}s")
            if inner_c is None: continue
            Hm, _ = cv.findHomography(X, inner_c, cv.RANSAC, 3.0)
            if Hm is not None: H_list.append(Hm)
            if len(H_list) >= 8:  # enough for stable solution
                break
        if len(H_list) < 4:
            # fallback: synthesize K from FOV if not enough views
            fx = fy = 0.5 * W / math.tan(math.radians(60.0)*0.5)
            cx, cy = W*0.5, H*0.5
            K = np.array([[fx, 0,  cx],[0, fy, cy],[0,0,1]], dtype=np.float64)
            print("[warn] Insufficient views. Using synthetic K (60° FOV).")
        else:
            K = estimate_K_from_homographies(H_list)
            print("[autoK] Estimated K:\n", K)
        dist = np.zeros(5, dtype=np.float64)
    map1, map2 = undistort_maps(K, dist, (W,H))

    # index -> light dir
    light_dir_by_m = {}
    total = len(unique_m)
    for idxk, j in enumerate(unique_m, 1):
        cap.set(cv.CAP_PROP_POS_FRAMES, j)
        ok, frame = cap.read()
        if not ok:
            continue
        und = cv.remap(frame, map1, map2, cv.INTER_LINEAR)
        small = und if det_scale == 1.0 else cv.resize(und, None, fx=det_scale, fy=det_scale, interpolation=cv.INTER_AREA)
        det_result = detect_marker_corners(
            small,
            th_block=args.th_block, th_C=args.th_C,
            min_area=args.min_area, max_ratio=args.max_ratio,
            dot_min_area=args.dot_min_area, dot_max_area=args.dot_max_area, dot_min_circ=args.dot_min_circ,
            area_scale=(det_scale*det_scale), dot_color=args.dot_color, dot_max_dist_frac=args.dot_max_dist_frac
        )

        if det_result is None:
            if args.progress_every > 0 and (idxk % args.progress_every == 0 or idxk == total):
                elapsed = time.time() - t0
                fps = idxk / max(1e-9, elapsed)
                print(f"[moving] {idxk}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s) - no marker detected")
            continue

        inner_c, outer_c, dot_pt = det_result
        if det_scale != 1.0:
            if inner_c is not None:
                inner_c = inner_c / det_scale
            if outer_c is not None:
                outer_c = outer_c / det_scale
            if dot_pt is not None:
                dot_pt = dot_pt / det_scale

        # homography plane->image (X->oc)
        Hmat, _ = cv.findHomography(X, inner_c, cv.RANSAC, 3.0)
        if Hmat is None:
            if args.progress_every > 0 and (idxk % args.progress_every == 0 or idxk == total):
                elapsed = time.time() - t0
                fps = idxk / max(1e-9, elapsed)
                print(f"[moving] {idxk}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")
            continue
        R, t = pose_from_homography(K, Hmat)
        L = -t  # light assumed co-located with camera center
        L = L / (np.linalg.norm(L) + 1e-12)
        light_dir_by_m[j] = L
        if args.debug and outer_c is not None and inner_c is not None:
            dbg2 = und.copy()
            # OUTER: thin green
            cv.polylines(dbg2, [outer_c.astype(int).reshape(-1,1,2)], True, (0,255,0), 1, cv.LINE_AA)
            for p in outer_c.astype(int):
                cv.circle(dbg2, tuple(p), 4, (0,200,0), 1, cv.LINE_AA)
            # INNER: thick cyan
            cv.polylines(dbg2, [inner_c.astype(int).reshape(-1,1,2)], True, (255,255,0), 3, cv.LINE_AA)
            for p in inner_c.astype(int):
                cv.circle(dbg2, tuple(p), 5, (255,255,0), -1)
            # Dot (if any)
            if dot_pt is not None:
                cv.circle(dbg2, tuple(dot_pt.astype(int)), 6, (0,0,255), -1)
            cv.imwrite(str(out/"debug"/f"outer_inner_{j:06d}.png"), dbg2)
        # debug draw
        # dbg = und.copy()
        # for p in oc.astype(int):
        #     cv.circle(dbg, tuple(p), 6, (0,255,0), 2)
        # txt = f"j={j}  L=({L[0]:+.2f},{L[1]:+.2f},{L[2]:+.2f})"
        # cv.putText(dbg, txt, (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,50), 2, cv.LINE_AA)
        # cv.imwrite(str(out/"debug"/f"m_{j:06d}.png"), dbg)
        # if args.progress_every > 0 and (idxk % args.progress_every == 0 or idxk == total):
        #     elapsed = time.time() - t0
        #     fps = idxk / max(1e-9, elapsed)
        #     print(f"[moving] {idxk}/{total} frames  ({fps:.1f} fps, elapsed {elapsed:.1f}s)")

    cap.release()

    # Build L in the order of static frames (pairs sequence) with preallocation
    dtype = np.float16 if args.save_lights_dtype == "float16" else np.float32
    lights = np.empty((len(pairs), 3), dtype=dtype)
    miss = 0
    last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i, (si, _, mi, _) in enumerate(pairs):
        if mi in light_dir_by_m:
            v = light_dir_by_m[mi].astype(np.float32)
            last = v
        else:
            v = last
            miss += 1
        lights[i] = v.astype(dtype, copy=False)

    # Write kept_pairs.csv with only pairs where light_dir_by_m has the moving idx
    kept_path = out/args.emit_kept_pairs
    with open(kept_path, "w") as fcsv:
        fcsv.write("static_idx,static_ts,moving_idx,moving_ts\n")
        for (si,sts,mi,mts) in pairs:
            if mi in light_dir_by_m:
                fcsv.write(f"{si},{sts:.6f},{mi},{mts:.6f}\n")
    print(f"[moving] kept_pairs written: {kept_path}")

    t_save0 = time.time()
    np.save(out/"lights.npy", lights)
    t_save = time.time() - t_save0
    elapsed_total = time.time() - t0
    print(f"[moving] lights saved: {out/'lights.npy'} shape={lights.shape} dtype={lights.dtype}  missing={miss}")
    print(f"[moving] timings: save={t_save:.2f}s  total={elapsed_total:.1f}s; valid={len(light_dir_by_m)} / requested={len(unique_m)}")

if __name__ == "__main__":
    main()