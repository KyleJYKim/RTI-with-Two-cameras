#!/usr/bin/env python3
"""
Note:
    Fitting Core Logic
        After analysis, two datasets are given: 
         1. Images: I_i(x,y) where i = 1..N
         2. Lights: l_i = (u_i,v_i) with u_i^2 + v_i^2 ≤ 1.
        RTI fitting means learning a function:
         I(x,y,u,v).
        Crucially, this function is learned independently for each pixel (x,y), 
        and there is no spatial learning in classic RTI.
    
    PTM vs. RBF
        PTM (Polynomial Texture Maps)
        Model: I(u,v) = a_0 + a_1*u + a_2*v + a_3*u^2 + a_4*u*v + a_5*u^2
        That's a 2D quadratic polynomial.
        Training (per px): Solve least squares: min_a Sum_i(I_i - 𝚽(u_i,v_i)a)^2
        where 𝚽(u,v) = [1,u,v,u^2,u*v,v^2]
        => PTM assumes smooth reflectance, no sharp specularities.
        => Pretty neat-fast because it uses closed-form least squares and only 6 coefficients are required per pixel.
        
        RBF (Radial Basis Function)
        Model: I(u,v) = Sum_k=1^K {w_k * exp(-(||(u,v) - c_k||^2)/(2*sigma^2))}
        where c_k are typically centers that are all measured lights or a sub-sampled set.
        Training: Solve linear system: K @ w = I
        where K_ik = xp(-(||(u,v) - c_k||^2)/(2*sigma^2))
        => RBF assumes local smoothness, better model specularities, and dense angular coverage.
        => Slow because it takes large kernel matrix with heavy memory and done per pixel.
        
"""

import cv2 as cv
import numpy as np

def fit_ptm(images, lights):
    """
    images: (N, H, W)
    lights: (N, 2)
    returns: coeffs (H, W, 6)
    """
    
    N, H, W = images.shape
    
    u = lights[:, 0]
    v = lights[:, 1]
    Phi = np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], axis=1)   # (N, 6)
    
    # Precompute pseudo-inverse (shared for all pixels)
    Phi_pinv = np.linalg.pinv(Phi)     # (6, N)

    coeffs = np.zeros((H, W, 6), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            I = images[:, y, x]        # (N,)
            coeffs[y, x] = Phi_pinv @ I

    return coeffs

def eval_ptm(coeffs, u, v):
    H, W, _ = coeffs.shape
    Phi = np.array([1, u, v, u*u, u*v, v*v], dtype=np.float32)
    return np.tensordot(coeffs, Phi, axes=([2],[0]))


def main():
    
    images = np.load("analysis/images.npy")
    lights = np.load("analysis/lights.npy")
    
    coeffs = fit_ptm(images, lights)
    np.save("analysis/ptm_coeffs.npy", coeffs)
    
    #u, v = lights[i]
    u, v = 0, 0
    pred = eval_ptm(coeffs, u, v)
    #gt = images[i]
    cv.imshow("pred", pred)
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()