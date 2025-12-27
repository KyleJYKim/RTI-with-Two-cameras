#!/usr/bin/env python3
"""
fit_reflectance.py

Fit a per-pixel reflectance model from MLIC Y-only and light directions.

Models:
  - PTM-QUAD   : I(l) ≈ a0 + a1*lx + a2*ly + a3*lz + a4*lx^2 + a5*ly^2 + a6*lz^2
                                  + a7*lx*ly + a8*ly*lz + a9*lz*lx
  - RBF        : I(l) ≈ b + Σ_j w_j * exp(-||l - c_j||^2 / (2σ^2)), centers c_j from k-means on lights

Inputs:
  - --mlic_y    : path to mlic_y.npy  (H,W,1,L), float32 in [0,1], Y-only stack
  - --lights    : path to lights.npy (L,3) unit vectors
  - --kept      : optional kept_pairs.csv to subset frames consistently

Outputs:
  - .npz with coefficients + metadata:
      PTM:  coeffs: (H,W,1,10),  model='ptm_quadratic', C=1, mode='luma'
      RBF:  coeffs: (H,W,1,M+1) (first is bias), centers: (M,3), sigma, model='rbf', C=1, mode='luma'
      Optional embedding of U,V per-pixel means if available (uv_mean.npz)

Memory/perf:
  - Processes pixels in tiles (--tile 20000) to keep memory <~ GB.
  - Precomputes a projection matrix P = (Φ^T Φ + λI)^{-1} Φ^T used for all pixels.

Usage (PTM):
  python3 fit_reflectance.py --mlic_y analysis/out_static/mlic_y.npy \
      --lights analysis/out_moving/lights.npy \
      --kept   analysis/out_moving/kept_pairs.csv \
      --model ptm --lambda 1e-3 --tile 20000 --out rtimodel_ptm.npz

Usage (RBF):
  python3 fit_reflectance.py --mlic_y analysis/out_static_G/mlic_y.npy \
      --lights analysis/out_moving_G/lights.npy \
      --kept   analysis/out_moving_G/kept_pairs.csv \
      --model rbf --rbf_centers 64 --rbf_sigma 0.35 --lambda 1e-3 \
      --tile 15000 --out rtimodel_rbf_G.npz \
      --norm --norm_target 0.7 --norm_band 0.12 --norm_coin_frac 0.22 \
      --embed_ab --embed_stride 1 \
          
      # Optional embedding of uv_mean.npz if present in same folder
"""

import argparse, csv, time
from pathlib import Path
import numpy as np

def load_kept_indices(kept_csv):
    if kept_csv is None: return None
    idxs = []
    with open(kept_csv, "r") as f:
        next(f)  # header
        for line in f:
            si, sts, mi, mts = line.strip().split(",")
            idxs.append(int(si))
    return np.array(idxs, dtype=np.int64)

def subset_stack(mlic, lights, kept_idx):
    """
    mlic: (H,W,1,L), lights: (L',3)
    kept_idx: array of static frame *IDs* from kept_pairs.csv (not necessarily 0..L-1).
    Cases:
      1) kept_idx is None -> require lights.shape[0] == L.
      2) max(kept_idx) < L -> assume mlic is full timeline; index by kept_idx.
      3) otherwise -> mlic was likely already built from a mapping; align by common length K.
    """
    L = mlic.shape[-1]
    Lp = lights.shape[0]
    if kept_idx is None:
        if Lp != L:
            K = min(L, Lp)
            print(f"[warn] L mismatch without kept: truncating to K={K} (mlic={L}, lights={Lp})")
            return mlic[..., :K], lights[:K]
        return mlic, lights
    if kept_idx.size == 0:
        raise SystemExit("kept_idx is empty")
    if int(np.max(kept_idx)) < L:
        # Safe to index directly
        idx = kept_idx.astype(np.int64)
        return mlic[..., idx], lights[:idx.shape[0]]
    else:
        # We cannot index mlic by absolute IDs; align by length only
        K = min(L, Lp, kept_idx.shape[0])
        print(f"[warn] kept_idx max >= L (max={int(np.max(kept_idx))}, L={L}). "
              f"Assuming mlic already subsetted; aligning by length K={K}.")
        return mlic[..., :K], lights[:K]

def features_ptm(lights):
    lx, ly, lz = lights[:,0], lights[:,1], lights[:,2]
    # 10 features per light
    Phi = np.stack([
        np.ones_like(lx),
        lx, ly, lz,
        lx*lx, ly*ly, lz*lz,
        lx*ly, ly*lz, lz*lx
    ], axis=1).astype(np.float32)  # (L,10)
    return Phi

def rbf_design(lights, centers, sigma):
    """Phi: (L, M+1) with bias term 1; RBF on S^2 using Euclidean distance in R^3.
    Since ||l||=||c||=1, d^2(l,c) = 2 - 2*(l·c).
    """
    L = lights.shape[0]
    M = centers.shape[0]
    Phi = np.empty((L, M+1), dtype=np.float32)
    Phi[:,0] = 1.0
    dots = lights @ centers.T              # (L,M)
    d2 = 2.0 - 2.0 * dots                  # (L,M)
    Phi[:,1:] = np.exp(-0.5 * d2 / (sigma*sigma))
    return Phi

def kmeans_centers(lights, M, iters=20, seed=0):
    # simple kmeans on unit vectors in R^3
    rng = np.random.default_rng(seed)
    N = lights.shape[0]
    if M >= N:
        # fallback: unique lights
        return lights.copy()
    idx = rng.choice(N, size=M, replace=False)
    C = lights[idx].copy()
    for _ in range(iters):
        # assign
        dots = lights @ C.T  # cosine similarity since unit vectors
        a = np.argmax(dots, axis=1)  # nearest by cosine
        # update
        for j in range(M):
            grp = lights[a==j]
            if len(grp)==0: 
                # reseed empty cluster
                C[j] = lights[rng.integers(0,N)]
            else:
                v = grp.mean(axis=0)
                n = np.linalg.norm(v)+1e-12
                C[j] = v / n
    return C

def compute_P(Phi, lam):
    """Return P that solves min_B ||Phi B - Y||^2 + lam||B||^2.
    Uses solve instead of explicit inverse for stability.
    """
    A = Phi.T @ Phi
    K = A.shape[0]
    A.flat[::K+1] += lam
    # Solve A X = Phi^T for X (X is P)
    P = np.linalg.solve(A, Phi.T)
    return P

def fit_chunked(P, Y, tile):
    """
    P: (K,L)
    Y: (L,N)  N = number of pixels in this chunk (and optionally channels)
    returns B: (K,N)
    """
    return P @ Y  # matmul is fast; chunking handled by caller

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlic_y", required=True, help="path to mlic_y.npy (H,W,1,L) Y-only stack")
    ap.add_argument("--lights", required=True)
    ap.add_argument("--kept", default=None, help="kept_pairs.csv to subset frames")
    ap.add_argument("--model", choices=["ptm","rbf"], default="ptm")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--tile", type=int, default=20000, help="pixels per tile")
    # Photometric normalization options (luma only)
    ap.add_argument("--norm", action="store_true",
                    help="Enable per-frame photometric normalization using a white ring mask")
    ap.add_argument("--norm_target", type=float, default=0.7,
                    help="Target median value in [0,1] for the white ring patch")
    ap.add_argument("--norm_band", type=float, default=0.12,
                    help="Inner/outer band thickness as fraction of image size to define white ring (e.g., 0.12)")
    ap.add_argument("--norm_coin_frac", type=float, default=0.22,
                    help="Radius (as fraction of min(H,W)) of central circle to EXCLUDE (coin area)")
    # RBF opts
    ap.add_argument("--rbf_centers", type=int, default=64)
    ap.add_argument("--rbf_sigma", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    # New arguments for embedding
    ap.add_argument("--embed_ab", action="store_true", help="Embed mlic and lights into the saved model for A/B viewer mode")
    ap.add_argument("--embed_stride", type=int, default=1, help="Stride for subsampling when embedding A/B data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    mlic = np.load(args.mlic_y)        # (H,W,1,L), float16/32 in [0,1], Y-only from YUV
    lights = np.load(args.lights)      # (L,3)
    kept_idx = load_kept_indices(args.kept) if args.kept else None
    mlic, lights = subset_stack(mlic, lights, kept_idx)
    if mlic.dtype != np.float32:
        mlic = mlic.astype(np.float32, copy=False)
    np.clip(mlic, 0.0, 1.0, out=mlic)
    H, W, C, L = mlic.shape
    assert C == 1, f"Expected Y-only (C=1), got C={C}"
    print(f"[fit] mlic shape={mlic.shape}, lights shape={lights.shape}")

    # Optional photometric normalization across frames (luma only)
    if args.norm:
        print(f"[fit:norm] target={args.norm_target} band={args.norm_band} coin_frac={args.norm_coin_frac}")
        # Build a ring mask: white band near borders minus a central coin disk
        yy, xx = np.ogrid[:H, :W]
        band = float(args.norm_band)
        x_lo = int(band * W); x_hi = int((1.0 - band) * W)
        y_lo = int(band * H); y_hi = int((1.0 - band) * H)
        rect = (xx > x_lo) & (xx < x_hi) & (yy > y_lo) & (yy < y_hi)
        border = ~rect
        cy, cx = H * 0.5, W * 0.5
        r = float(args.norm_coin_frac) * min(H, W)
        coin = (yy - cy) * (yy - cy) + (xx - cx) * (xx - cx) <= r * r
        mask = border & (~coin)
        num_mask = int(mask.sum())
        if num_mask == 0:
            print("[fit:norm][warn] normalization mask is empty; skipping normalization")
        else:
            target = float(args.norm_target)
            for l in range(L):
                patch = mlic[..., l][mask]  # Y-only
                m = float(np.median(patch))
                if m > 1e-6:
                    g = target / m
                    mlic[..., l] *= g
            np.clip(mlic, 0.0, 1.0, out=mlic)
            print("[fit:norm] per-frame Y normalization applied")

    # build design matrix Φ (L x K)
    if args.model == "ptm":
        Phi = features_ptm(lights)   # (L,10)
        centers = None; sigma = None
    else:
        centers = kmeans_centers(lights, args.rbf_centers, seed=args.seed)
        sigma = float(args.rbf_sigma)
        Phi = rbf_design(lights, centers, sigma)  # (L, M+1)

    L2, K = Phi.shape[0], Phi.shape[1]
    assert L2 == L, "design matrix L must match #frames"
    print(f"[fit] model={args.model}, L={L}, K={K}, λ={args.lam}")

    # Precompute projection P = (Φ^T Φ + λI)^(-1) Φ^T => (K,L)
    P = compute_P(Phi, args.lam)
    print(f"[fit] precomputed P matrix: {P.shape}")

    # Prepare output coeffs
    coeffs = np.empty((H, W, C, K), dtype=np.float32)

    # Reshape Y = (L, N*C) chunked
    # Note: mlic is (H,W,C,L) -> we want Y_l = intensities across frames for all pixels and one channel
    # We'll process channel by channel to reduce memory
    total_pix = H * W
    tile = int(args.tile)
    t1 = time.time()

    for ch in range(C):
        t_ch = time.time()
        print(f"[fit] channel {ch+1}/{C} ...")
        # arrange as (L, total_pix)
        Y_full = mlic[:,:,ch,:].reshape(H*W, L).T  # (L, total_pix)
        for start in range(0, total_pix, tile):
            end = min(total_pix, start + tile)
            Y = Y_full[:, start:end]          # (L, n)
            B = fit_chunked(P, Y, tile)       # (K, n)
            coeffs.reshape(-1, C, K)[start:end, ch, :] = B.T
        # free
        del Y_full
        print(f"[fit] channel {ch+1} done in {time.time()-t_ch:.1f}s")

    elapsed = time.time() - t1
    meta = dict(model=("ptm_quadratic" if args.model=="ptm" else "rbf"),
                K=K, L=L, H=H, W=W, C=C, mode="luma",
                lambda_reg=args.lam,
                centers=(centers if centers is None else centers.astype(np.float32)),
                sigma=(sigma if args.model=="rbf" else None))

    # Try to load per-pixel U/V means if present alongside the MLIC
    uv_path = Path(args.mlic_y).with_name("uv_mean.npz")
    U = V = None
    if uv_path.exists():
        try:
            uv = np.load(uv_path)
            U = uv.get("U", None)
            V = uv.get("V", None)
        except Exception:
            U = V = None

    if args.embed_ab:
        mlic_sub = mlic[..., ::args.embed_stride].astype(np.float16)
        lights_sub = lights[::args.embed_stride].astype(np.float32)
        if U is not None and V is not None:
            np.savez(args.out, coeffs=coeffs, mlic_y=mlic_sub, lights=lights_sub, U=U, V=V, **meta)
        else:
            np.savez(args.out, coeffs=coeffs, mlic_y=mlic_sub, lights=lights_sub, **meta)
    else:
        if U is not None and V is not None:
            np.savez(args.out, coeffs=coeffs, U=U, V=V, **meta)
        else:
            np.savez(args.out, coeffs=coeffs, **meta)
    print(f"[fit] saved {args.out}  coeffs.shape={coeffs.shape}  time={elapsed:.1f}s  total={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()