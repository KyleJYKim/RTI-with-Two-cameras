#!/usr/bin/env python3
"""
plot_lights.py

For examination!!!

Visualize light directions (u,v,w) from lights.npy on the dome,
optionally highlighting those kept in kept_pairs.csv.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_kept_indices(kept_csv):
    """Read static indices from kept_pairs.csv"""
    idxs = []
    with open(kept_csv, "r") as f:
        next(f)  # skip header
        for line in f:
            si, sts, mi, mts = line.strip().split(",")
            idxs.append(int(mi))  # use moving index (mi)
    return np.array(idxs, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lights", required=True, help="lights.npy from moving analysis")
    ap.add_argument("--kept", default=None, help="optional kept_pairs.csv for highlighting used lights")
    ap.add_argument("--out", default=None, help="optional path to save figure (png)")
    args = ap.parse_args()

    lights = np.load(args.lights)   # (L,3)
    print(f"[plot] loaded lights: {lights.shape}")

    lx, ly, lz = lights[:,0], lights[:,1], lights[:,2]
    norms = np.linalg.norm(lights, axis=1)
    print(f"[plot] norm stats: min={norms.min():.3f}, max={norms.max():.3f}")

    plt.figure(figsize=(6,6))

    # Plot all lights faintly
    plt.scatter(lx, ly, c=lz, cmap="viridis", s=5, alpha=0.3, label="all lights")

    if args.kept:
        kept_idx = load_kept_indices(args.kept)
        kept_idx = kept_idx[kept_idx < lights.shape[0]]
        plt.scatter(lx[kept_idx], ly[kept_idx], c=lz[kept_idx],
                    cmap="viridis", s=10, edgecolor="k", linewidth=0.3, label="kept")
        print(f"[plot] highlighted {len(kept_idx)} kept lights")

    # Unit circle boundary
    circle = plt.Circle((0,0), 1.0, color="k", fill=False, linestyle="--")
    plt.gca().add_artist(circle)
    plt.axis("equal")
    plt.xlabel("u (lx)")
    plt.ylabel("v (ly)")
    plt.title("Light directions (colored by lz)")
    plt.colorbar(label="lz (height)")
    plt.legend()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"[plot] saved {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()