#!/usr/bin/env python3
"""
Synchronize two videos (static/moving) using AUDIO cross-correlation and
emit a frame-to-frame mapping CSV that accounts for FPS mismatch.

Usage:
  python3 sync_by_audio.py \
    --static "./data/static_coin.mov" \
    --moving "./data/moving_coin.mov" \
    --out "./sync/mapping.csv" \
    --tmp_dir ".sync_tmp" \
    --sr 48000

Outputs:
  - mapping.csv with columns:
      static_idx, static_ts, moving_idx, moving_ts
    Static frame i should be paired with moving frame j.

Notes:
  - Requires ffmpeg/ffprobe in PATH.
  - Handles slight FPS differences by timestamp mapping, not naive 1:1 frame indices.
  - If one stream is longer, pairs within overlap only.
  - If audio has long silences, consider ensuring a brief clap/snap at start during capture.
"""

import argparse, os, sys, subprocess, json, math, csv, wave, contextlib
from pathlib import Path
import numpy as np
import cv2 as cv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True, help="path to static-camera video")
    ap.add_argument("--moving", required=True, help="path to moving-light video")
    ap.add_argument("--out", default="mapping.csv", help="output mapping CSV")
    ap.add_argument("--tmp_dir", default=".sync_tmp", help="temp dir for WAVs")
    ap.add_argument("--sr", type=int, default=48000, help="audio sample rate for extraction")
    ap.add_argument("--max_audio_lag", type=float, default=5.0, help="max expected lag (seconds) to search (+/-)")
    return ap.parse_args()

# ---------- ffmpeg helpers ----------

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout

def ffprobe_json(path):
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path)
    ]
    out = run_cmd(cmd)
    return json.loads(out)

def extract_audio_wav(inpath, outpath, sr=48000):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(inpath),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr),
        str(outpath)
    ]
    run_cmd(cmd)

# ---------- audio I/O ----------

def read_wav_mono_int16(path):
    with contextlib.closing(wave.open(str(path), 'rb')) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        if n_channels != 1 or sampwidth != 2:
            raise ValueError("Expected mono 16-bit PCM WAV")
        raw = wf.readframes(n_frames)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    return fr, x

# ---------- correlation & lag ----------

def normalized_cross_correlation(a, b, max_lag_samps=None):
    """
    Return lag (in samples) that maximizes correlation of a against b.
    Positive lag means 'a' must be shifted forward to match b (a starts earlier).
    """
    # normalize (zero-mean, unit-variance)
    a = a - np.mean(a)
    b = b - np.mean(b)
    a_std = np.std(a) + 1e-12
    b_std = np.std(b) + 1e-12
    a /= a_std
    b /= b_std

    # pick overlap length; use FFT-based xcorr for speed
    n = int(2 ** math.ceil(math.log2(len(a) + len(b) - 1)))
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    xcorr = np.fft.irfft(fa * np.conj(fb), n)

    # wrap so lags range is [-(len(b)-1), +(len(a)-1)]
    xcorr = np.concatenate([xcorr[-(len(b)-1):], xcorr[:len(a)]])
    lags = np.arange(-(len(b)-1), len(a))

    if max_lag_samps is not None:
        mask = (lags >= -max_lag_samps) & (lags <= max_lag_samps)
        lags = lags[mask]
        xcorr = xcorr[mask]

    best_idx = int(np.argmax(xcorr))
    return int(lags[best_idx]), float(xcorr[best_idx])

# ---------- fps & timestamps ----------

def get_video_fps_and_frames(path):
    # Prefer ffprobe; fallback to OpenCV if needed.
    try:
        info = ffprobe_json(path)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError("No video stream found")
        vs = vstreams[0]
        # r_frame_rate or avg_frame_rate
        fr_str = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "0/0"
        num, den = fr_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
        nb_frames = vs.get("nb_frames")
        if nb_frames is None or nb_frames == "N/A":
            # fallback to duration * fps
            dur = float(vs.get("duration") or info.get("format", {}).get("duration") or 0.0)
            n = int(round(dur * fps)) if fps > 0 and dur > 0 else 0
        else:
            n = int(nb_frames)
        if fps == 0.0 or n == 0:
            raise RuntimeError("Bad fps or frame count via ffprobe")
        return fps, n
    except Exception:
        cap = cv.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = cap.get(cv.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        if fps == 0.0 or n == 0:
            raise RuntimeError("Could not determine FPS/frames")
        return fps, n

# ---------- mapping ----------

def build_frame_mapping(fps_static, n_static, fps_moving, n_moving, lag_seconds):
    """
    Map each static frame index i to the closest moving frame index j.
    lag_seconds: positive if static audio must be shifted forward to match moving (static starts earlier).
    Returns list of tuples (i, t_s, j, t_m), keeping only pairs within overlap.
    """
    mapping = []
    for i in range(n_static):
        t_s = i / fps_static          # timestamp of static frame i
        t_m = t_s + lag_seconds       # corresponding moment in moving timeline
        j = int(round(t_m * fps_moving))
        if j < 0 or j >= n_moving:
            continue
        mapping.append((i, t_s, j, j / fps_moving))
    return mapping

def choose_frames_to_drop(mapping, n_static, n_moving):
    """
    Optional helper: if one stream is much faster, propose a set of indices to drop from the faster stream
    so the pairings become non-decreasing. Returns two sets: drop_static, drop_moving.
    """
    drop_static = set()
    drop_moving = set()
    last_i, last_j = -1, -1
    for i, _, j, _ in mapping:
        if i <= last_i:
            drop_static.add(i)  # out-of-order static (rare with our construction)
        if j <= last_j:
            drop_moving.add(j)  # indicates faster moving stream
        last_i, last_j = i, j
    return drop_static, drop_moving

# ---------- main ----------

def main():
    args = parse_args()
    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    wav_static = tmp / "static.wav"
    wav_moving = tmp / "moving.wav"

    print("[1/5] Extracting audio...")
    extract_audio_wav(args.static, wav_static, sr=args.sr)
    extract_audio_wav(args.moving, wav_moving, sr=args.sr)

    print("[2/5] Reading audio and computing lag...")
    sr_s, a = read_wav_mono_int16(wav_static)
    sr_m, b = read_wav_mono_int16(wav_moving)
    if sr_s != args.sr or sr_m != args.sr:
        print(f"[warn] WAV SR mismatch (got {sr_s}, {sr_m}) vs requested {args.sr}; proceeding.")

    max_lag_samps = int(args.max_audio_lag * args.sr)
    lag_samps, corr = normalized_cross_correlation(a, b, max_lag_samps=max_lag_samps)
    lag_seconds = lag_samps / float(args.sr)

    # Interpretation:
    # We computed lag such that shifting STATIC forward by lag aligns with MOVING.
    # Positive lag_seconds => static starts earlier in time.
    print(f"[lag] best lag: {lag_seconds:+.4f} s  (NCC={corr:.3f})  "
          f"[+ => shift STATIC forward]")

    print("[3/5] Probing FPS & frame counts...")
    fps_s, n_s = get_video_fps_and_frames(args.static)
    fps_m, n_m = get_video_fps_and_frames(args.moving)
    print(f"[info] static: fps={fps_s:.6f}, frames={n_s}")
    print(f"[info] moving: fps={fps_m:.6f}, frames={n_m}")

    print("[4/5] Building frame mapping...")
    mapping = build_frame_mapping(fps_s, n_s, fps_m, n_m, lag_seconds)
    if not mapping:
        print("Error: empty mapping (no temporal overlap?)")
        sys.exit(2)

    # Optional: frame-drop suggestions to enforce monotonic pairing (usually small)
    drop_s, drop_m = choose_frames_to_drop(mapping, n_s, n_m)
    if drop_s or drop_m:
        print(f"[hint] Proposed drops -> static:{len(drop_s)} moving:{len(drop_m)} "
              "(only if you need strictly monotonic pairs)")

    print(f"[5/5] Writing CSV: {args.out}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["static_idx", "static_ts", "moving_idx", "moving_ts"])
        for i, ts, j, tm in mapping:
            w.writerow([i, f"{ts:.6f}", j, f"{tm:.6f}"])

    print("[done] Frame mapping saved.")
    print("      Example: static frame i -> moving frame j; "
          "use timestamps if you resample streams later.")

if __name__ == "__main__":
    main()