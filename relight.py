import numpy as np
import cv2 as cv

DOME_SIZE = 300
    
def colorize(Y, U, V):
    Y8 = np.clip(Y * 255, 0, 255).astype(np.uint8)
    U8 = np.clip(U * 255, 0, 255).astype(np.uint8)
    V8 = np.clip(V * 255, 0, 255).astype(np.uint8)
    
    # yuv to bgr
    # Y(uv)
    yuv = cv.merge([Y8, U8, V8])
    bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
    
    return bgr

# def evaluate_ptm(coeffs, u, v): # coeffs: (H, W, 6), u, v: scalars
#     a = [0,0,0,0,0,0]
#     for i in range(6):
#         a[i] = coeffs[:, :, i]
    
#     Y = a[0]*u*u + a[1]*v*v + a[2]*u*v + a[3]*u + a[4]*v + a[5]
    
#     return Y # (H, W) float32

def eval_ptm(coeffs, u, v):
    #H, W, _ = coeffs.shape
    Phi = np.array([1, u, v, u*u, u*v, v*v], dtype=np.float32)
    return np.tensordot(coeffs, Phi, axes=([2],[0]))
    
def get_light_direction(x, y, W, H):
    u = (x / W) * 2.0 - 1.0
    v = (y / H) * 2.0 - 1.0
    if u*u + v*v > 1.0: # not on the hemisphere
        print(f"Light direction: {u, v, u*u + v*v}")
        return None
    return u, v

def draw_dome(lights, size, uv=None):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size //2
    r = size // 2 - 5
    
    # hemisphere boundary
    cv.circle(image, (cx, cy), r, (200, 200, 200), 2)
    
    for l in lights:
        u, v = l
        x = int(cx + u * r)
        y = int(cy + -v * r)
        cv.circle(image, (x, y), 6, (125, 125, 125), -1)
    
    if uv is not None:
        u, v = uv
        x = int(cx + u * r)
        y = int(cy + -v * r)
        cv.circle(image, (x, y), 6, (0,0, 255), -1)
    
    return image

def on_dome_mouse(event, x, y, flags, param):
    global mouse_uv
    size = param
    cx, cy = size // 2, size // 2
    r = size // 2 - 5
    
    dx = (x - cx) / r
    dy = (y - cy) / r
    
    if dx*dx + dy*dy <= 1.0:
        mouse_uv = (dx, -dy)
        print(f"on dome: {dx, dy}")
        
def main():
    # Predict Y(x,y | u,v)
    # Combine with stored U_avg and V_avg
    # Convert YUV to RGB, 
    # i.e., RGB(x,y,u,v) = YUV(Y(x,y,u,v), U_avg(x,y), V_avg(x,y))
    
    images = np.load("analysis/images.npy") # (N, H, W), float32 [0,1]
    lights = np.load("analysis/lights.npy") # (N, 2)
    U_avg = np.load("analysis/U_avg.npy")   # (H, W), float32 [0,1]
    V_avg = np.load("analysis/V_avg.npy")   # (H, W), float32 [0,1]
    
    ptm_coeffs = np.load("analysis/ptm_coeffs.npy") # (H, W, 6)
    print(f"Shape of PTM Coeffs: {ptm_coeffs.shape}")
    
    #H, W = images.shape[1:]
    
    global mouse_uv
    mouse_uv = None
    cv.namedWindow("Relighting")
    cv.namedWindow("Dome")
    cv.setMouseCallback("Dome", on_dome_mouse, DOME_SIZE)
    
    # Manual Test with the captured images and lights
    # for i in range(len(lights)):
        
    #     u, v = lights[i]
    #     img = images[i]
    #     print(u,v)
    #     relit_Y = eval_ptm(ptm_coeffs, u, v)
    #     relit_Y = np.clip(relit_Y, 0.0, 1.0)        # clamp luminance
    #     relit_RGB = colorize(relit_Y, U_avg, V_avg)
    #     cv.imshow("Relighting", relit_RGB)
    #     cv.imshow("Lighting", img)
    #     cv.waitKey(0)
    
    
    print(lights.min(axis=0), lights.max(axis=0))
    
    while True:
        if mouse_uv is not None:
            print(mouse_uv)
            u, v = mouse_uv
            
            # per-pixel gain
            # def enhance(Y, gain=1.5):
            #     Yc = Y - Y.mean()
            #     return np.clip(Y.mean() + gain * Yc, 0, 1)
            
            
            relit_Y = eval_ptm(ptm_coeffs, u, v)
            relit_Y = np.clip(relit_Y, 0.0, 1.0)        # clamp luminance
            relit_RGB = colorize(relit_Y, U_avg, V_avg)
            
            # Visualization
            # cv.imshow("Relit Y", np.clip(relit_Y * 255, 0, 255).astype(np.uint8))
            # cv.imshow("Relit RGB", relit_RGB)
            cv.imshow("Relighting", relit_RGB)
        
        
        cv.imshow("Dome", draw_dome(lights, DOME_SIZE, mouse_uv))
        
        key = cv.waitKey(1)
        if key == ord('q'): break
    
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()