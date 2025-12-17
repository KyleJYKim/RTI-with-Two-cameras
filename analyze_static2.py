import sys
import json
import cv2 as cv
import numpy as np

SKIP_FRMS = 100
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
            
        # Homography
        dst_pts = np.array( [ [0,0], [512,0], [512,512], [0,512] ] ) 
        #H = cv.getPerspectiveTransform(quad, dst_pts)
        H = cv.findHomography(quad, dst_pts)
        warped = cv.warpPerspective(gray, H[0], (512, 512))
        
        return quad, warped, contours

    return None, None, contours

            

def main():
    
    save_path = None # "./data/static/sobel_imgs"
    analysis_vid = cv.VideoCapture("./data/static/coin.MOV")
    K, dist = load_calibration_data("./calib/static/intrinsics.json")
    
    video_length = int(analysis_vid.get(cv.CAP_PROP_FRAME_COUNT))
    running = True
    frame_count = 0
    frame_num = 1
    
    while running:
        frame_count += 1
        success, frame = analysis_vid.read()
        if not success: break

        frame = cv.undistort(frame, K, dist)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5,5), 0) # smoother background
        
        outer_quad, outer_warped, contours = detect_edges(gray)
        if outer_quad is None:
            continue
        
        margin = int(0.01 * 512)   # crop 1% of width because of outer square detection!
        roi = outer_warped[margin:-margin, margin:-margin]
        inner_quad, inner_warped, contours = detect_edges(roi)
        if inner_quad is None:
            continue



        # Visualization: draw quad
        inner_warped = cv.cvtColor(inner_warped, cv.COLOR_GRAY2BGR)
        vis_frame = inner_warped.copy()
        #cv.polylines(vis_frame, [quad.astype(int)], True, (0,255,0), 2)
        cv.polylines(vis_frame, contours, True, (0,255,0), 2)
        cv.imshow(f"frame", vis_frame)
        #cv.imshow(f"frame", inner_warped)

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