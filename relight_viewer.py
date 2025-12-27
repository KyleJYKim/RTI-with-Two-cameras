#!/usr/bin/env python3
"""
Load a fitted RTI model (PTM-Quad or RBF) and interactively relight with a virtual light.
Controls:
  - Mouse drag: move light on a hemisphere (azimuth φ, elevation θ)
  - Keys:
      q/esc : quit
      S     : save current render to PNG
      s     : swap X and Y
      1/2   : change elevation step
      [/]   : change azimuth step
      x/y/z : flip respective axis of light vector
      r     : rotate +90° in-plane (x'=-y, y'=x)
      t     : rotate -90° in-plane (x'=y, y'=-x)
      a     : toggle A/B mode (use captured light k)
      j/k   : prev/next captured light index (A/B mode)
      g     : toggle Ground Truth vs Model (A/B mode)
  - Dome window:
      Drag the handle on the circle. The 2D position (x,y) is mapped to hemisphere coords
      (u=azimuth°, v=elevation°). Relighting updates in real time.

Usage:
  python3 relight_viewer.py \
    --model rtimodel_ptm__G.npz \
    --dome \
    --display_gamma 2.2
"""
import argparse, time
import numpy as np
import cv2 as cv
from pathlib import Path

# ---------------- Dome control helpers ----------------

def clamp_unit_disk(x, y):
    r2 = x*x + y*y
    if r2 <= 1.0:
        return x, y
    r = np.sqrt(r2)
    return x/r, y/r

def angles_from_xy(x, y):
    """Given x,y in [-1,1] on the dome projection, return (azim_deg, elev_deg) and lvec.
    Mapping: l = (x, y, +sqrt(max(0,1-x^2-y^2))) with y upward.
    Note: image Y grows downward, so we'll flip sign when converting from pixels.
    """
    x, y = clamp_unit_disk(float(x), float(y))
    lz = float(np.sqrt(max(0.0, 1.0 - x*x - y*y)))
    lx, ly = float(x), float(y)
    # azimuth in degrees [0,360)
    az = np.degrees(np.arctan2(ly, lx)) % 360.0
    # elevation (0..90): 90 - polar
    elev = 90.0 - np.degrees(np.arccos(lz))
    lvec = np.array([lx, ly, lz], dtype=np.float32)
    return az, elev, lvec

def xy_from_angles(azim_deg, elev_deg):
    """Inverse of angles_from_xy. Given azimuth and elevation, return x,y in [-1,1]."""
    th = np.deg2rad(90.0 - np.clip(elev_deg, 0.0, 90.0))
    ph = np.deg2rad(azim_deg)
    lx = np.sin(th)*np.cos(ph)
    ly = np.sin(th)*np.sin(ph)
    # ignore lz here; project to unit disk
    return float(lx), float(ly)

def draw_dome(canvas, cx, cy, rad, x, y, az, elev):
    """Draws the dome circle and handle at (x,y) in [-1,1], with text of (x,y,az,elev)."""
    h, w = canvas.shape[:2]
    canvas[:] = (30, 30, 30)
    # circle
    cv.circle(canvas, (cx, cy), rad, (80, 80, 80), 2, cv.LINE_AA)
    # crosshair
    cv.line(canvas, (cx-rad, cy), (cx+rad, cy), (60,60,60), 1, cv.LINE_AA)
    cv.line(canvas, (cx, cy-rad), (cx, cy+rad), (60,60,60), 1, cv.LINE_AA)
    # valid region mask (none needed visually beyond circle)
    # handle position (flip y for pixels)
    px = int(round(cx + x*rad))
    py = int(round(cy - y*rad))
    cv.circle(canvas, (px, py), 8, (180, 220, 250), -1, cv.LINE_AA)
    # text
    cv.putText(canvas, f"x={x:+.2f} y={y:+.2f}  az={az:6.1f}°  el={elev:5.1f}°", (12, 24),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv.LINE_AA)
    cv.putText(canvas, "Drag handle to move light", (12, h-12),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv.LINE_AA)

def phi_vector_ptm(l):
    lx, ly, lz = l
    return np.array([
        1.0,
        lx, ly, lz,
        lx*lx, ly*ly, lz*lz,
        lx*ly, ly*lz, lz*lx
    ], dtype=np.float32)  # (10,)

def phi_vector_rbf(l, centers, sigma):
    M = centers.shape[0]
    v = np.empty(M+1, dtype=np.float32); v[0] = 1.0
    d2 = np.sum((centers - l[None,:])**2, axis=1)
    v[1:] = np.exp(-0.5 * d2 / (sigma*sigma))
    return v

def render(model, coeffs, lvec, centers=None, sigma=None):
    H, W, C, K = coeffs.shape
    if model == "ptm_quadratic":
        phi = phi_vector_ptm(lvec)             # (10,)
    else:
        phi = phi_vector_rbf(lvec, centers, sigma)  # (M+1,)
    # (H*W*C,K) @ (K,) -> (H*W*C,)
    out = (coeffs.reshape(-1, K) @ phi).reshape(H, W, C)
    # Return linear RGB float in [0,1]; caller will apply display gamma and convert to BGR8
    return np.clip(out, 0.0, 1.0)

def yuv_to_rgb(yuv):
    """Convert YUV image (H,W,3) with Y,U,V in [0,1] to RGB in [0,1]."""
    Y = yuv[...,0]
    U = yuv[...,1] - 0.5
    V = yuv[...,2] - 0.5
    R = Y + 1.402 * V
    G = Y - 0.344136 * U - 0.714136 * V
    B = Y + 1.772 * U
    rgb = np.stack([R,G,B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".npz from fit_reflectance.py")
    ap.add_argument("--elev", type=float, default=40.0, help="initial elevation (deg)")
    ap.add_argument("--azim", type=float, default=45.0, help="initial azimuth (deg)")
    ap.add_argument("--dome", action="store_true", help="Show RTI dome controller window")
    ap.add_argument("--dome_size", type=int, default=420, help="Dome window size (pixels)")
    ap.add_argument("--display_gamma", type=float, default=2.2,
                    help="Gamma to apply for display (inverse EOTF). Use 1.0 to disable.")
    args = ap.parse_args()

    data = np.load(args.model, allow_pickle=True)
    coeffs = data["coeffs"]  # (H,W,3,K) or (H,W,1,K)
    model = str(data["model"])
    # Normalize model if it's an ndarray (e.g. bytes/object array)
    if isinstance(model, np.ndarray):
        try:
            model = str(model.item())
        except Exception:
            model = str(model)
    # Optional for A/B comparison: ground-truth warped frames and lights
    mlic_gt = None
    mlic_y_gt = None
    if "mlic" in data.files:
        try:
            mlic_gt = data["mlic"]  # expected shape (H,W,3,L), linear [0,1]
        except Exception:
            mlic_gt = None
    if "mlic_y" in data.files:
        try:
            mlic_y_gt = data["mlic_y"]  # expected shape (H,W,1,L), linear [0,1]
        except Exception:
            mlic_y_gt = None
    lights_gt = None
    if "lights" in data.files:
        try:
            lights_gt = data["lights"].astype(np.float32)
        except Exception:
            lights_gt = None
    ab_enabled = False
    H, W, C, K = coeffs.shape
    is_luma = (C == 1)
    uv_mean = None
    # Try loading U,V from uv_mean.npz in same directory as model if luma only
    if is_luma:
        model_path = Path(args.model)
        uv_mean_path = model_path.parent / "uv_mean.npz"
        if uv_mean_path.exists():
            try:
                uv_data = np.load(str(uv_mean_path))
                U = uv_data.get("U", None)
                V = uv_data.get("V", None)
                if U is not None and V is not None and U.shape == (H, W) and V.shape == (H, W):
                    uv_mean = np.stack([U, V], axis=-1)  # (H,W,2)
                    print(f"[viewer] Loaded uv_mean from {uv_mean_path}")
            except Exception:
                uv_mean = None
    # Determine A/B mode availability and GT data shape matching
    if mlic_gt is not None and lights_gt is not None and mlic_gt.shape[0]==H and mlic_gt.shape[1]==W:
        ab_enabled = True
    elif mlic_y_gt is not None and lights_gt is not None and mlic_y_gt.shape[0]==H and mlic_y_gt.shape[1]==W:
        ab_enabled = True
    if not ab_enabled:
        print("[viewer] A/B mode disabled (model lacks embedded mlic/lights or size mismatch)")
    # centers/sigma exist only for RBF models. Be robust to None / object arrays.
    centers = None
    if "centers" in data.files:
        c = data["centers"]
        if isinstance(c, np.ndarray) and c.size > 0:
            centers = c
    sigma = None
    if "sigma" in data.files:
        s = data["sigma"]
        if s is None:
            sigma = None
        elif isinstance(s, np.ndarray):
            if s.size == 0:
                sigma = None
            else:
                try:
                    sigma = float(s.item())
                except Exception:
                    sigma = None
        else:
            try:
                sigma = float(s)
            except Exception:
                sigma = None

    print(f"[viewer] model={model} coeffs={coeffs.shape} is_luma={is_luma}")

    # Track light via spherical angles (deg)
    elev = args.elev  # 0..90
    azim = args.azim  # 0..360
    elev_step = 3.0
    azim_step = 5.0

    # Light-vector mapping controls (default: flip Z)
    M = np.eye(3, dtype=np.float32)
    flip = np.array([1.0, 1.0, -1.0], dtype=np.float32)

    def print_map():
        print(f"[map] flip={flip.tolist()}  M=\n{M}")

    # --- Dome controller window state ---
    dome_on = bool(args.dome)
    dome_name = "RTI Dome"
    if dome_on:
        D = max(240, int(args.dome_size))
        dome = np.zeros((D, D, 3), np.uint8)
        cx = cy = D//2
        rad = int(0.46 * D)
        # start from given azim/elev; compute x,y on disk
        xh, yh = xy_from_angles(azim, elev)
        dragging = False
        def on_mouse(event, mx, my, flags, userdata):
            nonlocal xh, yh, azim, elev, dragging
            # convert pixels to normalized x,y in [-1,1]
            dx = (mx - cx) / float(rad)
            dy = (cy - my) / float(rad)  # flip sign for y-up
            if event == cv.EVENT_LBUTTONDOWN:
                dragging = True
            if event == cv.EVENT_MOUSEMOVE and dragging:
                dx, dy = clamp_unit_disk(dx, dy)
                az, el, _ = angles_from_xy(dx, dy)
                xh, yh = dx, dy
                azim, elev = az, el
            if event == cv.EVENT_LBUTTONUP:
                dragging = False
        cv.namedWindow(dome_name, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(dome_name, on_mouse)

    def l_from_angles(elev_deg, azim_deg):
        th = np.deg2rad(90.0 - max(0.0, min(90.0, elev_deg)))  # polar from +Z
        ph = np.deg2rad(azim_deg)
        lx = np.sin(th)*np.cos(ph)
        ly = np.sin(th)*np.sin(ph)
        lz = np.cos(th)
        return np.array([lx,ly,lz], dtype=np.float32)

    win = "RTI Relighter"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, min(900, W), min(900, H))

    last = None
    ab_on = False
    ab_show_gt = False  # False: show model; True: show GT
    ab_k = 0
    while True:
        if ab_on and ab_enabled:
            # Use exact captured light; do NOT apply mapping flips/rotations here
            lvec = lights_gt[ab_k]
        else:
            if dome_on:
                _, _, lvec = angles_from_xy(xh, yh)
            else:
                lvec = l_from_angles(elev, azim)
            # Apply flips and in-plane mapping (ensure float32)
            lvec = (lvec * flip).astype(np.float32)
            lvec = (M @ lvec).astype(np.float32)
        if last is None or not np.allclose(lvec, last):
            print(f"[light] {lvec}")
            last = lvec.copy()
        if ab_on and ab_enabled and ab_show_gt:
            # Ground-truth warped frame at index ab_k (assumed linear RGB or Y in [0,1])
            if is_luma and mlic_y_gt is not None:
                y_lin = np.clip(mlic_y_gt[..., ab_k], 0.0, 1.0)  # (H,W,1)
                if uv_mean is not None:
                    yuv = np.concatenate([y_lin, uv_mean], axis=-1)  # (H,W,3)
                    lin = yuv_to_rgb(yuv)
                else:
                    lin = np.repeat(y_lin, 3, axis=2)
            elif mlic_gt is not None:
                lin = np.clip(mlic_gt[..., ab_k], 0.0, 1.0)
            else:
                # fallback to model rendering if no GT data
                lin = render(model, coeffs, lvec, centers, sigma)
        else:
            lin = render(model, coeffs, lvec, centers, sigma)   # linear RGB or Y in [0,1]
            if is_luma:
                # lin shape (H,W,1)
                if uv_mean is not None:
                    yuv = np.concatenate([lin, uv_mean], axis=-1)  # (H,W,3)
                    lin = yuv_to_rgb(yuv)
                else:
                    lin = np.repeat(lin, 3, axis=2)
        g = float(args.display_gamma)
        srgb = np.power(lin, 1.0/g, dtype=lin.dtype) if (g and g != 1.0) else lin
        vis = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        bgr = cv.cvtColor(vis, cv.COLOR_RGB2BGR)
        hud = bgr.copy()
        if ab_on and ab_enabled:
            mode = "GT" if ab_show_gt else "MODEL"
            txt = f"A/B [{mode}] k={ab_k}/{(lights_gt.shape[0]-1)}"
            cv.rectangle(hud, (8,8), (8+len(txt)*10+6, 32), (0,0,0), -1)
            cv.putText(hud, txt, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
        bgr_disp = hud
        cv.imshow(win, bgr_disp)
        if dome_on:
            draw_dome(dome, cx, cy, rad, xh, yh, azim, elev)
            cv.imshow(dome_name, dome)
        key = cv.waitKey(20) & 0xFF
        if key in (27, ord('q')): break
        elif key == ord('S'):
            outp = Path(args.model).with_suffix("")
            ts = int(time.time())
            p = f"{outp}_e{int(elev)}_a{int(azim)}_{ts}.png"
            cv.imwrite(p, bgr)
            print("[saved]", p)
        elif key == ord('1'):
            elev_step = max(1.0, elev_step - 1.0)
        elif key == ord('2'):
            elev_step += 1.0
        elif key == ord('['):
            azim_step = max(1.0, azim_step - 1.0)
        elif key == ord(']'):
            azim_step += 1.0
        # arrow keys: ← → azim, ↑ ↓ elev
        elif key == 81:  # left
            azim -= azim_step
            if dome_on:
                xh, yh = xy_from_angles(azim, elev)
        elif key == 83:  # right
            azim += azim_step
            if dome_on:
                xh, yh = xy_from_angles(azim, elev)
        elif key == 82:  # up
            elev = min(90.0, elev + elev_step)
            if dome_on:
                xh, yh = xy_from_angles(azim, elev)
        elif key == 84:  # down
            elev = max(0.0, elev - elev_step)
            if dome_on:
                xh, yh = xy_from_angles(azim, elev)
        elif key == ord('x'):
            flip[0] *= -1; print_map()
        elif key == ord('y'):
            flip[1] *= -1; print_map()
        elif key == ord('z'):
            flip[2] *= -1; print_map()
        elif key == ord('s'):  # swap X and Y
            M = M @ np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=np.float32); print_map()
        elif key == ord('r'):  # +90° in-plane
            M = M @ np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float32); print_map()
        elif key == ord('t'):  # -90° in-plane
            M = M @ np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.float32); print_map()
        elif key == ord('a'):
            if ab_enabled:
                ab_on = not ab_on
                print(f"[A/B] {'enabled' if ab_on else 'disabled'}")
            else:
                print("[A/B] not available: model lacks embedded mlic/lights")
        elif key == ord('g'):
            if ab_on and ab_enabled:
                ab_show_gt = not ab_show_gt
                print(f"[A/B] show {'GT' if ab_show_gt else 'MODEL'}")
        elif key == ord('j'):
            if ab_on and ab_enabled:
                ab_k = max(0, ab_k - 1)
        elif key == ord('k'):
            if ab_on and ab_enabled:
                ab_k = min(lights_gt.shape[0]-1, ab_k + 1)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()