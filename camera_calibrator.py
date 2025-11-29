#!/usr/bin/env python3
"""
Usage:
    For static camera:
        python3 camera_calibrator.py \
        --video "./data/static_calibration.mov" \
        --out "./calib/static.yaml" \
        --preview "./calib/static_undistorted_preview.mp4" \
        --show

    For moving camera:
        python3 camera_calibrator.py \
        --video "./data/moving_calibration.mov" \
        --out "./calib/moving.yaml" \
        --preview "./calib/moving_undistorted_preview.mp4" \
        --show

Notes:
- rows, cols = number of inner corners on the chessboard (OpenCV convention).
- square = square size in mm.
- Output: calib.yaml with K, dist (k1,k2,p1,p2,k3), image_size, reprojection_error.
- Saves an undistorted preview video for sanity check.

Spec alignment:
- Picks ≥20 frames with successful detections, spaced across the video.
- Uses cv2.calibrateCamera and cv2.undistort.
"""
import numpy as np
import cv2 as cv
import argparse, sys, math, os
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows",       type=int,   default=6,   help="inner corners (rows)")
    ap.add_argument("--cols",       type=int,   default=9,   help="inner corners (cols)")
    ap.add_argument("--square",     type=float, default=1.0, help="square size (mm)")
    ap.add_argument("--min_frames", type=int,   default=20,  help="minimum successful frames")
    ap.add_argument("--max_frames", type=int,   default=120, help="hard cap on processed frames")
    ap.add_argument("--video",      type=str,   default="./data_G/cam1/calibration.mov")
    ap.add_argument("--out",        type=str,   default="./calib_G/cam1_calib.yaml")
    ap.add_argument("--preview",    type=str,   default="./calib_G/cam1_undistorted_preview.mp4")
    ap.add_argument("--show", action="store_true", help="interactive preview while detecting")
    return ap.parse_args()

def build_object_points(rows, cols, square):
    # Chessboard model in Z=0 plane
    objp = np.zeros((rows*cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)  # (x=cols, y=rows)
    objp[:,:2] = grid * square
    return objp  # shape (N,3)

def evenly_spaced_indices(n_total, n_pick):
    if n_pick >= n_total:
        return np.arange(n_total)
    # space indices across [0, n_total-1]
    return np.unique(np.round(np.linspace(0, n_total-1, n_pick)).astype(int))

def detect_corners(gray, pattern_size):
    # Try modern SB (more robust), fall back to classic
    flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    ok, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if not ok:
        ok, corners = cv.findChessboardCorners(gray, pattern_size,
                    flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            # refine with sub-pixel
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
            cv.cornerSubPix(gray, corners, (5,5), (-1,-1), term)
    return ok, corners

def collect_detections(cap, rows, cols, min_frames, max_frames, show=False):
    pattern_size = (cols, rows)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
    # Heuristic: attempt up to max_frames spread across the video + first/last
    n_attempt = min(max_frames, max(min_frames*2, 60))
    cand_idxs = evenly_spaced_indices(total_frames, n_attempt).tolist()
    force_idxs = {0, max(0, total_frames-1)}
    for i in force_idxs: 
        if i not in cand_idxs: cand_idxs.append(i)
    cand_idxs = sorted(set(cand_idxs))

    objp = build_object_points(rows, cols, 1.0)  # unit square; scale later
    objpoints, imgpoints, used_idxs = [], [], []

    H, W = None, None
    for fi in cand_idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: continue
        if H is None:
            H, W = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ok, corners = detect_corners(gray, pattern_size)
        if ok:
            objpoints.append(objp.copy())   # unit spacing
            imgpoints.append(corners.reshape(-1,1,2))
            used_idxs.append(fi)
            if show:
                vis = frame.copy()
                cv.drawChessboardCorners(vis, pattern_size, corners, True)
                cv.imshow("detections", vis)
                if cv.waitKey(1) == 27: break
        if len(objpoints) >= min_frames:
            # keep going a bit more for coverage, but not too much
            if len(objpoints) >= min_frames + 10:
                break

    if show:
        cv.destroyAllWindows()

    return objpoints, imgpoints, (W, H), used_idxs

def scale_object_points(objpoints, square_size):
    scaled = []
    for op in objpoints:
        s = op.copy()
        s[:, :2] *= square_size
        scaled.append(s)
    return scaled

def calibrate(objpoints, imgpoints, imsize):
    W, H = imsize
    K_init = np.array([[0.8*W, 0, W/2],
                       [0, 0.8*W, H/2],
                       [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros(5)  # k1,k2,p1,p2,k3
    flags = (cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K3)  # start simple; we’ll re-run enabling K3
    # First pass (stable start)
    ret1, K1, dist1, rvecs1, tvecs1 = cv.calibrateCamera(
        objpoints, imgpoints, (W,H), K_init, dist_init, flags=flags)
    # Second pass enabling 5-param (k1,k2,p1,p2,k3)
    flags2 = 0  # free all
    ret2, K2, dist2, rvecs2, tvecs2 = cv.calibrateCamera(
        objpoints, imgpoints, (W,H), K1, dist1, flags=flags2)
    return ret2, K2, dist2, rvecs2, tvecs2

def save_yaml(path, K, dist, imsize, reproj_err):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write("image_width", int(imsize[0]))
    fs.write("image_height", int(imsize[1]))
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist.reshape(-1,1))
    fs.write("reprojection_error", float(reproj_err))
    fs.release()

def undistort_preview(in_video, out_video, K, dist):
    cap = cv.VideoCapture(in_video)
    if not cap.isOpened(): return
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(out_video, fourcc, fps, (W,H))
    if not writer.isOpened():
        print("[warn] could not open writer for preview:", out_video)
        return
    # No getOptimalNewCameraMatrix per spec; direct undistort [oai_citation:2‡G3DCV2024_FinalProject.pdf](file-service://file-5s5bcj13aqQ8c24z5ShBbB)
    map1, map2 = cv.initUndistortRectifyMap(K, dist, None, K, (W,H), cv.CV_16SC2)
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    show_every = max(1, n_frames // 100)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        und = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)
        writer.write(und)
        i += 1
        if i % show_every == 0:
            sys.stdout.write(f"\r[preview] {i}/{n_frames}")
            sys.stdout.flush()
    writer.release()
    cap.release()
    print("\n[preview] saved:", out_video)

def main():
    args = parse_args()
    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: cannot open video:", args.video)
        sys.exit(1)

    objp_u, imgp, imsize, used = collect_detections(
        cap, args.rows, args.cols, args.min_frames, args.max_frames, show=args.show)
    cap.release()

    if len(objp_u) < args.min_frames:
        print(f"Error: only {len(objp_u)} successful detections (< {args.min_frames}). "
              "Ensure good chessboard contrast and coverage.")
        sys.exit(2)

    # Scale object points to real square size
    objp = scale_object_points(objp_u, args.square)

    print(f"[info] Using {len(objp)} frames at indices: {used[:8]}{' ...' if len(used)>8 else ''}")
    print(f"[info] Image size: {imsize}")

    reproj_err, K, dist, rvecs, tvecs = calibrate(objp, imgp, imsize)
    print("[result] Reprojection RMSE (pixels):", reproj_err)
    print("[result] K:\n", K)
    print("[result] dist [k1 k2 p1 p2 k3]:\n", dist.ravel())

    save_yaml(args.out, K, dist, imsize, reproj_err)
    print("[saved]", args.out)

    if args.preview:
        undistort_preview(args.video, args.preview, K, dist)

if __name__ == "__main__":
    main()