import sys
import json
import cv2 as cv
import numpy as np

SKIP_FRMS = 5000
SCALE = 9   # kernel size
K = 0.04     # optimal: 0.04 – 0.15
THRESH = 0.01
NMS_WINSIZE = 13
    
def load_calibration_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "K" in data and "distortion" in data:
        dist = np.array(data["distortion"], dtype=np.float64).ravel()
        K = np.array(data["K"], dtype=np.float64)
    else:
        raise ValueError("No calibration data found.")
    return K, dist

def detect_edges(gray):
    #_, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #gray = cv.addWeighted(frame, 0.5, gray, 0.5, 0) # sharpen edges
    edges = cv.Canny(gray, 50, 150)
    #edges = cv.morphologyEx(edges, cv.MORPH_OPEN, np.ones((1,1), np.uint8))

    # dilate a bit to close gaps
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    height, width = gray.shape[:2]
    img_area = height * width
    quad = None
    for contour in sorted(contours, key=cv.contourArea, reverse=True):
        area = cv.contourArea(contour)
        if area < 0.05 * img_area: break    # too small to be a board

        # Find the inner square
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True) # count edges
        if len(approx) == 4 and cv.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(np.float32)

            # consistent order: [tl, tr, br, bl]
            s = pts.sum(axis=1)
            d = np.diff(pts, axis=1).ravel()
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(d)]
            bl = pts[np.argmax(d)]
            quad = np.array([tl, tr, br, bl], dtype=np.float32)
        
        if quad is None or quad.shape != (4, 2): break
        
        # Homography
        dst_pts = np.array([[0,0], [512,0], [512,512], [0,512]]) 
        
        H = cv.findHomography(quad, dst_pts)
        
        if H[0] is None: break
        
        warped = cv.warpPerspective(gray, H[0], (512, 512))
        
        return quad, warped, contours

    return None, None, contours


def set_fiducial_orientation(warped, fiducial_pos, size=512):
    """
    Ensure fiducial lies in top-left quadrant.
    Returns rotated image, rotated fiducial, and rotation matrix R_f.
    """
    x, y = fiducial_pos
    cx, cy = size // 2, size // 2

    # determine quadrant
    if x < cx and y < cy:
        rot_k = 0
    elif x >= cx and y < cy:
        rot_k = 1   # rotate 90 CCW
    elif x >= cx and y >= cy:
        rot_k = 2   # rotate 180
    else:
        rot_k = 3   # rotate 270 CCW

    warped_rot = np.rot90(warped, k=rot_k)
    fid_rot = np.array([x, y], dtype=np.float32)

    for _ in range(rot_k):
        fid_rot = np.array([fid_rot[1], size - fid_rot[0]], dtype=np.float32)

    theta = rot_k * (np.pi / 2.0)
    R_f = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [ 0,              0,             1]
    ], dtype=np.float32)

    return warped_rot, fid_rot, R_f

def main():
    
    save_path = None # "./data/static/sobel_imgs"
    # analysis_vid = cv.VideoCapture("./data/moving/coin_shifted.MOV")
    # K, dist = load_calibration_data("./calib/moving/intrinsics.json")
    analysis_vid = cv.VideoCapture("./data_G/cam2/coin1_shifted.mov")
    K, dist = load_calibration_data("./data_G/cam2/intrinsics.json")
    
    video_length = int(analysis_vid.get(cv.CAP_PROP_FRAME_COUNT))
    running = True
    frame_count = 0
    frame_num = 1
    
    while running:
        frame_count += 1
        success, frame = analysis_vid.read()
        if not success: break

        height, width = frame.shape[:2]
        optimal_K, optimal_frame = cv.getOptimalNewCameraMatrix(K, dist, (width, height), alpha=0)

        frame = cv.undistort(frame, K, dist, None, optimal_K)

        x, y, w, h = optimal_frame
        frame = frame[y:y+h, x:x+w]

        #frame = cv.undistort(frame, K, dist)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 0) # smoother background
        
        outer_quad, outer_warped, contours = detect_edges(gray)
        if outer_quad is None:
            continue
        
        print("outer")
        
        margin = int(0.01 * 512)   # crop 1% of width because of outer square detection!
        roi = outer_warped[margin:-margin, margin:-margin]
        inner_quad, inner_warped, contours = detect_edges(roi)
        if inner_quad is None:
            continue
            
        print("inner")
        
        
        vis_frame = roi.copy()
        vis_frame = cv.cvtColor(vis_frame, cv.COLOR_GRAY2BGR)
        cv.polylines(vis_frame, contours, True, (0,255,0), 2)
        # cv.imshow(f"frame", vis_frame)
        # cv.waitKey(0)
        
        
        
        
        # Masking for fiducial
        mask = np.full_like(outer_warped, 255, dtype=np.uint8)
        # fill entire outer square
        #cv.fillConvexPoly(mask, outer_quad.astype(np.int32), 255)
        # remove inner square
        cv.fillConvexPoly(mask, inner_quad.astype(np.int32), 0)
        masked = cv.bitwise_and(outer_warped, outer_warped, mask=mask)
        masked = cv.GaussianBlur(masked, (9,9), 1.5)
        masked = cv.bitwise_not(masked)
        
        # Find fiducial
        circles = cv.HoughCircles(masked, cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=50, minRadius=8,maxRadius=60)
        
    
        
        if circles is not None:
            circles = np.round(circles[0]).astype(np.int32)
            x_f, y_f, r_f = circles[np.argmax(circles[:,2])]
        else:
            continue
        
        print("circle")
        
        # Visualization
        vis_frame = cv.cvtColor(outer_warped, cv.COLOR_GRAY2BGR)
        vis_frame = vis_frame.copy()
        cv.polylines(vis_frame, [inner_quad.astype(int)], True, (0,0,255), 2)
        #cv.polylines(vis_frame, contours, True, (0,255,0), 2)
        cv.circle(vis_frame, (x_f, y_f), r_f, (0,255,0), 2)
        cv.imshow(f"frame", vis_frame)
        cv.waitKey(10)
        
        
        outer_warped, fid_xy, R_f = set_fiducial_orientation(outer_warped, (x_f, y_f), 512)
        
        cx, cy = 256, 256
        R_hemi = 256.0

        u = (fid_xy[0] - cx) / R_hemi
        v = (cy - fid_xy[1]) / R_hemi

        if u*u + v*v > 1.0:
            continue

        w = np.sqrt(1.0 - u*u - v*v)
        l_h = np.array([u, v, w], dtype=np.float32)
        
        Hmat = H[0]
        Hnorm = np.linalg.inv(optimal_K) @ Hmat
        r1 = Hnorm[:,0]
        r2 = Hnorm[:,1]
        r1 /= np.linalg.norm(r1)
        r2 /= np.linalg.norm(r2)
        r3 = np.cross(r1, r2)
        R_h = np.stack([r1, r2, r3], axis=1)

        l_surface = R_h.T @ (R_f.T @ l_h)

        print("Light direction (surface frame):", l_surface)
        

        if save_path is not None:
            cv.imwrite(f"{save_path}/frame_{frame_num:05d}.png", vis_frame)

        if SKIP_FRMS + frame_num >= video_length: break

        for _ in range(SKIP_FRMS): analysis_vid.read()
        frame_num += SKIP_FRMS

        keyp = cv.waitKey(1)
        running = keyp != 113  # Press q to exit

    cv.destroyAllWindows()
    analysis_vid.release()

if __name__ == "__main__":
    main()