import numpy as np
import cv2 as cv

SKIP_FRMS = 0
SCALE = 9   # kernel size
K = 0.04     # optimal: 0.04 – 0.15
THRESH = 0.01
NMS_WINSIZE = 13
H_SIZE = 512
INNER_SIZE = 256

def line_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    norm = np.sqrt(a*a + b*b)
    return a/norm, b/norm, c/norm

def fit_line_pca(points):
    xm, ym = points.mean(axis=0)
    Q = points - np.array([xm, ym])
    
    _, _, Vt = np.linalg.svd(Q)
    # smallest singular value
    normal = Vt[-1]
    a, b = normal
    c = -a*xm - b*ym
    norm = np.sqrt(a*a + b*b)
    return a/norm, b/norm, c/norm
    
def ransac_line(points, n_iters=500, eps=2.0, min_inliers=50):
    best_inliers = []
    best_model = None
    N = len(points)
    
    for _ in range(n_iters):
        # indices of random 2 points
        i, j = np.random.choice(N, 2, replace=False)
        model = line_from_two_points(points[i], points[j])
        a, b, c = model
        
        # perpendicular distance = |a*x_i + b*y_i + c|
        dists = np.abs(a*points[:,0] + b*points[:,1] + c)
        inliers = points[dists < eps]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model
        
    if best_model is None or len(best_inliers) < min_inliers:
        return None, None
    # refine using PCA on inliers
    refined_model = fit_line_pca(best_inliers)
    return refined_model, best_inliers
        
def intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    A = np.array([[a1, b1], [a2, b2]])
    C = np.array([-c1, -c2])
    return np.linalg.solve(A, C)

def detect_rectangle(frame):
    edges = cv.Canny(frame, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    lines = []
    height, width = frame.shape[:2]
    img_area = height * width

    for points in sorted(contours, key=cv.contourArea, reverse=True):
        # reshape contour from (N,1,2) to (N,2)
        points = points.reshape(-1, 2)
        area = cv.contourArea(points)
        if area < 0.02 * img_area: break    # too small to be a board
        
        while len(points) > 50:
                    
            line, inliers = ransac_line(points)
            if line is None:
                break
            
            lines.append(line)
            # points are removed once used
            points = np.array([p for p in points if p.tolist() not in inliers.tolist()])
        
            # theta = np.arctan2(-b, a)  # line direction
            # lines.append((a, b, c, theta))

    if len(lines) < 4:
        return None, None, None, None
    
    print(f"Four LINES: {lines}")
    # Visualization
    #vis_frame1 = cv.cvtColor(frame_org.copy(), cv.COLOR_GRAY2BGR)
    vis_frame = frame.copy()
    cv.polylines(vis_frame, [points.astype(np.int32)], True, (0,0,255), 2)
    cv.imshow(f"contour", vis_frame)
    cv.waitKey(1)
    
    quad = np.ndarray([4])
    group1 = []
    group2 = []

    a, b, _ = lines[0]
    ref_angle =  np.arctan2(-b, a)

    for line in lines:
        a, b, _ = line
        angle =  np.arctan2(-b, a)
        diff = np.abs(np.sin(angle - ref_angle))  # sin handles pi ambiguity

        if diff < 0.7:   # roughly parallel
            group1.append(line)
        else:
            group2.append(line)
    
    # sort parallel pairs
    group1 = sorted(group1, key=lambda l: l[2])
    group2 = sorted(group2, key=lambda l: l[2])

    # decide which group is vertical
    a1, b1, _ = group1[0]
    if abs(a1) > abs(b1):
        vertical = group1
        horizontal = group2
    else:
        vertical = group2
        horizontal = group1
    
    quad = np.array([
        intersect_lines(vertical[0], horizontal[0]),
        intersect_lines(vertical[1], horizontal[0]),
        intersect_lines(vertical[1], horizontal[1]),
        intersect_lines(vertical[0], horizontal[1]),
    ])
    
    #if quad is None or quad.shape != (4, 2): break
    
    # Homography
    dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
    
    # Find teh projective transform that maps the detected square in the image to a perfectly aligned square of the size.
    H_mat, _ = cv.findHomography(quad, dst_pts)
    
    if H_mat is None: return None, None, None, None
    
    warped = cv.warpPerspective(frame, H_mat, (H_SIZE, H_SIZE))
    
    return quad, warped, H_mat, contours
    

def detect_square(frame):
    gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0) # smoother background
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
    # 1
    edges = cv.Canny(gray, 50, 150)
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    # 2
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # extract edge points
    ys, xs = np.where(edges > 0)
    points = np.stack([xs, ys], axis=1)
    #detect_line(points)

    
    height, width = gray.shape[:2]
    img_area = height * width
    quad = None
    for contour in sorted(contours, key=cv.contourArea, reverse=True):
        area = cv.contourArea(contour)
        if area < 0.02 * img_area: break    # too small to be a board

        # Find the inner square
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.05 * peri, True) # count edges
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
        
        
    
        # Visualization
        # vis_frame1 = gray = cv.cvtColor(gray.copy(), cv.COLOR_GRAY2BGR)
        # cv.polylines(vis_frame1, contours, True, (0,0,255), 2)
        # cv.imshow(f"gray_frame outer", vis_frame1)
        # cv.waitKey(1)
        # if quad is None or quad.shape != (4, 2):
        #     cv.waitKey(0)
        # else: 
        #     cv.waitKey(1)
        
        
        if quad is None or quad.shape != (4, 2): break
        
        # Homography
        dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
        
        # Find teh projective transform that maps the detected square in the image to a perfectly aligned square of the size.
        H_mat, _ = cv.findHomography(quad, dst_pts)
        
        if H_mat is None: break
        
        warped = cv.warpPerspective(frame, H_mat, (H_SIZE, H_SIZE))
        
        return quad, warped, H_mat, contours

    return None, None, None, contours

def main():
    analysis_vid1 = cv.VideoCapture("./data_G/cam2/coin1_synced.mov")
    
    W = int(analysis_vid1.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(analysis_vid1.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    outer_quad1 = None
    inner_quad1 = None
    fid_x1, fid_y1, fid_r1 = None, None, None
    running = True
    
    # static cam detection only requires once and then all the frames share the same detections (outer, inner, fiducial)
    while running:
        keyp = cv.waitKey(1)
        running = keyp != 113  # Press q to exit
        success1, frame1 = analysis_vid1.read()
        if not success1: break

        margin = int(0.01 * 512)   # crop 1% of width due to detection of outer square!
        outer_roi1 = frame1[margin:-margin, margin:-margin]
        outer_roi1 = cv.cvtColor(outer_roi1.copy(), cv.COLOR_BGR2GRAY)
        outer_roi1 = cv.GaussianBlur(outer_roi1, (5,5), 0) # smoother background
        
        roi_outer_quad1, outer_warped1, _, _ = detect_rectangle(outer_roi1)
        if roi_outer_quad1 is None: continue
        outer_quad1 = roi_outer_quad1 + np.array([margin, margin], dtype=np.float32) # back to org coords
        
        print("Detected static outer square")
        
        # Visualization
        vis_frame1 = frame1.copy()
        cv.polylines(vis_frame1, [outer_quad1.astype(int)], True, (0,0,255), 2)
        cv.imshow(f"static_frame outer", vis_frame1)
        cv.waitKey(1)
        
        
        margin = int(0.1 * 512)   # crop
        inner_roi1 = outer_warped1[margin:-margin, margin:-margin]
        #inner_roi1 = cv.adaptiveThreshold(inner_roi1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 3)
        
        ##### LET'S DETECT ONCE FOR MOVING INNER SQUARE SAME AS STATIC SQUARES!!!
        
        roi_inner_quad1, roi_inner_warped1, _, _ = detect_rectangle(inner_roi1)
        if roi_inner_quad1 is None: continue
        inner_quad1 = roi_inner_quad1 + np.array([margin, margin], dtype=np.float32) # back to org coords
        
        print("Detected inner square")
        
        
        # Visualization
        vis_frame1 = cv.cvtColor(inner_roi1.copy(), cv.COLOR_GRAY2BGR)
        cv.polylines(vis_frame1, [roi_inner_quad1.astype(int)], True, (0,0,255), 2)
        cv.imshow(f"static_frame inner", vis_frame1)
        cv.waitKey(1)
        
        if (outer_quad1 is not None and inner_quad1 is not None) and (fid_x1 is not None and fid_y1 is not None and fid_r1 is not None): break
        
        #if 1 + frame_num >= video_length1: break
    
if __name__ == "__main__":
    main()