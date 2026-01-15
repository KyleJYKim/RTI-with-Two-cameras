import sys
import math
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SKIP_FRMS = 10
SCALE = 9   # kernel size
K = 0.04     # optimal: 0.04 – 0.15
THRESH = 0.01
NMS_WINSIZE = 13
H_SIZE = 512
INNER_SIZE = 256
    
def load_calibration_data(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "K" in data and "distortion" in data:
            dist = np.array(data["distortion"], dtype=np.float64).ravel()
            K = np.array(data["K"], dtype=np.float64)
        else:
            return None, None
    except FileNotFoundError:
        return None, None
    return K, dist

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

def order_quad(quad): # tl, tr, br, bl
    quad = np.asarray(quad, dtype=np.float32)
    centroid = quad.mean(axis=0)

    # angle around centroid
    angles = np.arctan2(quad[:,1] - centroid[1], quad[:,0] - centroid[0])
    quad = quad[np.argsort(angles)]

    # ensure TL first: TL has minimal (x + y)
    idx = np.argmin(quad[:,0] + quad[:,1])
    quad = np.roll(quad, -idx, axis=0)

    return quad

def detect_square(frame_org, graying=True):
    frame = frame_org.copy()
    if graying:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (5,5), 0) # smoother background
        frame = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
    edges = cv.Canny(frame, 50, 150)
    edges = cv.dilate(edges, np.ones((3,3), np.uint8), iterations=1)    # dilate a bit to close gaps
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
    
    #print(f"Four LINES: {lines}")
    # Visualization
    #vis_frame1 = cv.cvtColor(frame_org.copy(), cv.COLOR_GRAY2BGR)
    # vis_frame = frame.copy()
    # cv.polylines(vis_frame, [points.astype(np.int32)], True, (0,0,255), 2)
    # cv.imshow(f"contour", vis_frame)
    # cv.waitKey(1)
    
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
    # while image quad is tr-tl-bl-br, destination points is tl-tr-br-bl, so need to be ordered.
    quad = order_quad(quad)
    dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
    # Find the projective transform that maps the detected square in the image to a perfectly aligned square of the size.
    H_mat, _ = cv.findHomography(quad, dst_pts)
    
    if H_mat is None: return None, None, None, None
    
    warped = cv.warpPerspective(frame_org, H_mat, (H_SIZE, H_SIZE))
    
    return quad, warped, H_mat, contours

def fit_circle_svd(points):
    n = len(points)
    x = points[:,0]
    y = points[:,1]
    
    A = np.vstack([x*x + y*y, x, y, np.ones(n)]).T
    _, s, v = np.linalg.svd(A)
    #print(s)

    # right singular vector with smallest singular value
    # this minimizes algebraic error - standard least-squares circle fit.
    params = v[-1]
    #print(params)

    xc = -params[1]/(2*params[0])
    yc = -params[2]/(2*params[0])
    # print(xc)
    # print(yc)
    r = np.sqrt(np.abs((params[1]**2 + params[2]**2)/(4*params[0]**2) - params[3]/params[0]))
    
    return xc, yc, r

def detect_fiducial(frame, inner_quad):
    gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9,9), 0) # smoother background
    _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = np.full_like(gray, 255, dtype=np.uint8)  # starting with the full white is more effective
    # fill entire outer square
    #cv.fillConvexPoly(mask, outer_quad.astype(np.int32), 255)
    # mask inner square
    cv.fillConvexPoly(mask, inner_quad.astype(np.int32), 0)
    masked = cv.bitwise_and(gray, gray, mask=mask)
    masked = cv.GaussianBlur(masked, (9,9), 1.5)
    masked = cv.bitwise_not(masked)
    edges = cv.Canny(masked, 80, 160)
    
    # Find the fiducial
    ys, xs = np.where(edges > 0)
    if len(xs) < 50: return None, None, None
    
    points = np.column_stack([xs, ys])
    xc, yc, r = fit_circle_svd(points)
    
    # print(f"RADIUS SIZE: {r}")
    # Visualization
    # vis_frame1 = masked.copy()
    # cv.circle(vis_frame1, (int(xc), int(yc)), int(r), (0,255,0), 2)
    # cv.imshow(f"frame fiducial", vis_frame1)
    # cv.waitKey(1)
    
    if r < 10 or r > 20: return None, None, None    # around 15
    
    return int(xc), int(yc), int(r)
    
def set_frame_orientation(warped, quad, fiducial_pos, size=H_SIZE):
    """
    Because the warped images are not guaranteed to face the same direction,
    this here ensures fiducial to position in top-left quadrant,
    and returns rotated image, rotated fiducial, and rotation matrix R_f.
    R_f is required because we need to know how much the original homography is rotated 
    for calculating the coordinate of the light source, i.e., the moving camera position!
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
    quad_rot = quad.copy().astype(np.float32)
    fid_rot = np.array([x, y], dtype=np.float32)
    for _ in range(rot_k): 
        quad_rot = np.column_stack([quad_rot[:,1], size - quad_rot[:,0]])
        fid_rot = np.array([fid_rot[1], size - fid_rot[0]], dtype=np.float32)
        
    return warped_rot, quad_rot, fid_rot, rot_k


def set_frame_orientation2(warped, quad, fiducial_pos, size=H_SIZE):
    """
    Reorders the quad so that the corner containing the fiducial mark is at top-left,
    and rotates the image correspondingly for visualization purposes.
    
    Returns:
        warped_rot: rotated image for visualization
        quad_rot: reordered quad with fiducial corner at top-left
        fid_rot: fiducial position in rotated image
        rot_k: number of 90-degree CCW rotations applied
    """
    x, y = fiducial_pos
    quad = quad.astype(np.float32)

    # --- 1. enforce CCW order around centroid ---
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:,1] - center[1], quad[:,0] - center[0])
    quad_ccw = quad[np.argsort(angles)]

    # --- 2. pick TL corner using fiducial ---
    dists = np.linalg.norm(quad_ccw - np.array([x, y]), axis=1)
    tl_idx = np.argmin(dists)

    # rotate so TL is first → [TL, TR, BR, BL]
    quad_rot = np.roll(quad_ccw, -tl_idx, axis=0)

    # --- 3. ensure correct orientation (no mirroring) ---
    # check signed area (should be CCW)
    v1 = quad_rot[1] - quad_rot[0]
    v2 = quad_rot[3] - quad_rot[0]
    if np.cross(v1, v2) < 0:
        quad_rot = quad_rot[[0,3,2,1]]

    # --- 4. visualization-only rotation (keep fiducial in TL quadrant) ---
    cx, cy = size // 2, size // 2
    fx, fy = quad_rot[0]
    if fx < cx and fy < cy:
        rot_k = 0
    elif fx >= cx and fy < cy:
        rot_k = 1
    elif fx >= cx and fy >= cy:
        rot_k = 2
    else:
        rot_k = 3

    if rot_k == 0:
        warped_rot = warped
    else:
        warped_rot = np.rot90(warped, k=rot_k)

    quad_img = quad_rot.copy()
    fid_rot = np.array([x, y], dtype=np.float32)
    for _ in range(rot_k):
        quad_img = np.column_stack([quad_img[:,1], size - quad_img[:,0]])
        fid_rot = np.array([fid_rot[1], size - fid_rot[0]], dtype=np.float32)

    return warped_rot, quad_img, fid_rot, rot_k



def synthesize_K(W, H, fov_deg=60.0):
    # pinhole guess: fx = fy = 0.5*W / tan(FOV/2), cx=W/2, cy=H/2
    fx = fy = 0.5 * W / math.tan(math.radians(fov_deg)*0.5)
    cx, cy = W*0.5, H*0.5
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist

def main():
    """
    Note: 
        - Appended digit 1 and 2 to variables indicate static cam and moving cam, respectively.
           e.g., K1 and K2 mean K for static cam and K for moving cam.
        - Images will be rotated to position the fiducial mark on top-left.
    
    The main sequence for each frame follows:
        1. Apply calibration.
        2. In the 1st loop, detect the outer and inner square of static cam once and for all the others.
        3. In the 2nd loop:
            - Get the warped image of static cam;
            - Detect and get the outer quad and warped image of moving cam;
            - Detect the fiducial marks of static and moving cams by masking the warped images with the inner_quad1 detected in the 1st loop
              (the inner_quad1 applies to both warped images because they are in the same size, hence inner_quad2 = inner_quad1);
            - Rotate the warped images according to the position of the fiducial marks and get the rotation amounts;
            - Crop the static cam image for saving;
            - Calculate the light direction with the rotation data;
            - Save and repeat.
    """
    save_path = None # "./data/static/sobel_imgs"
    analysis_vid1 = cv.VideoCapture("./data_G/cam1/coin1_synced.mov")
    K1, dist1 = load_calibration_data("./data_G/cam1/intrinsics.json")
    analysis_vid2 = cv.VideoCapture("./data_G/cam2/coin1_synced.mov")
    K2, dist2 = load_calibration_data("./data_G/cam2/intrinsics.json")
    
    W = int(analysis_vid1.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(analysis_vid1.get(cv.CAP_PROP_FRAME_HEIGHT))
    if K1 is None:
        K1, dist1 = synthesize_K(W, H)  # heuristic intrinsics, zero distortion
        print("[warning] No calib provided/found. Using synthetic K and zero distortion.")
    
    # both outer and inner square of static frames, 
    # and inner square of moving frames are captured only once
    outer_quad1 = None
    inner_quad1 = None
    fid_x1, fid_y1, fid_r1 = None, None, None
    running = True
    
    # Note: static cam detection only requires once and then all the frames share the same detections (outer, inner, fiducial)
    while running:
        keyp = cv.waitKey(1)
        running = keyp != 113  # Press q to exit
        success1, frame1 = analysis_vid1.read()
        if not success1: break

        frame1 = cv.undistort(frame1.copy(), K1, dist1)

        margin = int(0.01 * H_SIZE)   # crop 1% of width due to detection of outer square!
        roi1 = frame1[margin:-margin, margin:-margin]
        
        roi_outer_quad1, outer_warped1, _, _ = detect_square(roi1)
        if roi_outer_quad1 is None: continue
        outer_quad1 = roi_outer_quad1 + np.array([margin, margin], dtype=np.float32) # back to org coords
        
        print("Detected static outer square")
        # Visualization
        # vis_frame1 = outer_warped1.copy()
        # cv.polylines(vis_frame1, [outer_quad1.astype(int)], True, (0,0,255), 2)
        # cv.imshow(f"static frame initial outer", vis_frame1)
        # cv.waitKey(0)
        
        margin = int(0.1 * H_SIZE)   # crop
        
        roi1 = outer_warped1[margin:-margin, margin:-margin]
        roi_inner_quad1, roi_inner_warped1, _, _ = detect_square(roi1, False)
        if roi_inner_quad1 is None: continue
        inner_quad1 = roi_inner_quad1 + np.array([margin, margin], dtype=np.float32) # back to org coords
        
        print("Detected static inner square")
        # Visualization
        # vis_frame1 = roi1.copy()
        # cv.polylines(vis_frame1, [roi_inner_quad1.astype(int)], True, (0,0,255), 2)
        # cv.imshow(f"static frame initial inner", vis_frame1)
        # cv.waitKey(0)
        
        margin = int(0.02 * H_SIZE)   # crop
        roi1 = outer_warped1[margin:-margin, margin:-margin]
        # inflate inner quad for inner square masking
        center = np.mean(inner_quad1, axis=0)
        scale = 1.07  # inflation factor 7%
        roi_inner_quad1 = center + scale * (inner_quad1 - center)
        
        fid_x1, fid_y1, fid_r1 = detect_fiducial(roi1, roi_inner_quad1)
        if (fid_x1 or fid_y1 or fid_r1) is None: 
            print("NO static fiducial mark")
            continue
        fid_x1 += margin
        fid_y1 += margin
        
        print("Detected static fiducial mark")
        # Visualization
        # vis_frame1 = outer_warped1.copy()
        # cv.polylines(vis_frame1, [inner_quad1.astype(int)], True, (0,0,255), 2)
        # cv.circle(vis_frame1, (fid_x1, fid_y1), fid_r1, (0,255,0), 2)
        # cv.imshow(f"static frame initial fiducial", vis_frame1)
        # cv.waitKey(1)
        
        if (outer_quad1 is not None and inner_quad1 is not None) and (fid_x1 is not None and fid_y1 is not None and fid_r1 is not None): break
        
        if save_path is not None:
            cv.imwrite(f"{save_path}/frame_{frame_num:05d}.png", vis_frame1)

        if 1 + frame_num >= video_length1: break
    
    
    video_length1 = int(analysis_vid1.get(cv.CAP_PROP_FRAME_COUNT))
    video_length2 = int(analysis_vid2.get(cv.CAP_PROP_FRAME_COUNT))
    inner_quad2 = inner_quad1   # assuming after warp, the inner square sizes are same.
    running = True
    frame_cnt = 1
    frame_num = 1
    all_images = []         # from static cam, gray
    all_lights = []         # from moving cam
    all_U = []              # for colorizing
    all_V = []              # for colorizing
    mvn_sqr_fail_cnt = 0    # for debugging
    mvn_fid_fail_cnt = 0    # for debugging
    hom_fid_fail_cnt = 0
    
    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots()
    sc = ax.scatter([], [])
    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Light directions (u, v)")
    ax.grid(True)
    
    while running:
        success1, frame1 = analysis_vid1.read()
        success2, frame2 = analysis_vid2.read()
        if not (success1 and success2): break
        
        print(f"FRAME NUMBER: {frame_num:05d}/{video_length1:05d}")
        print(f"FRAME COUNT: {frame_cnt:05d}")
        
        frame1 = cv.undistort(frame1.copy(), K1, dist1)
        frame2 = cv.undistort(frame2.copy(), K2, dist2)

        margin = int(0.01 * H_SIZE)   # crop 1% of width due to detection of outer square!
        
        # static cam uses the same quad for consistency!
        frame1[margin:-margin, margin:-margin]
        dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
        H_outer1, _ = cv.findHomography(outer_quad1, dst_pts)
        if H_outer1 is not None:
            outer_warped1 = cv.warpPerspective(frame1, H_outer1, (H_SIZE, H_SIZE))
            
            # moving cam search for lighting!
            roi2 = frame2[margin:-margin, margin:-margin]
            outer_quad2, outer_warped2, H_outer2, _ = detect_square(roi2)
            if outer_quad2 is None or H_outer2 is None:
                print("Failed moving outer square")
                cv.imwrite(f"./analysis/moving_sqr_failed/frame_{frame_num:05d}.png", roi2)
                mvn_sqr_fail_cnt += 1
            else:
                print("Detected moving outer square")
                # Visualization
                # vis_frame1 = frame1.copy()
                # cv.polylines(vis_frame1, [outer_quad1.astype(int)], True, (0,0,255), 2)
                # vis_frame2 = frame2.copy()
                # cv.polylines(vis_frame2, [outer_quad2.astype(int)], True, (0,0,255), 2)
                # cv.imshow(f"static frame outer", vis_frame1)
                # cv.imshow(f"moving frame outer", vis_frame2)
                # keyp = cv.waitKey(1)
                
                #margin = int(0.01 * H_SIZE)   # crop 1.2%
                # well.. this is only for visualization
                # roi1 = outer_warped1[margin:-margin, margin:-margin]
                # roi_inner_quad1, roi_inner_warped1, _, _ = detect_square(roi1, False)
                # if roi_inner_quad1 is None: continue
                #inner_quad1 = roi_inner_quad1 + np.array([margin, margin], dtype=np.float32) # back to org coords
                
                #roi2 = outer_warped2[margin:-margin, margin:-margin]
                
                # Visualization
                vis_frame2 = outer_warped2.copy()
                cv.polylines(vis_frame2, [inner_quad2.astype(int)], True, (0,0,255), 2)
                cv.imshow(f"moving frame inner", vis_frame2)
                keyp = cv.waitKey(1)
                
                margin = int(0.02 * H_SIZE)   # crop
                roi2 = outer_warped2[margin:-margin, margin:-margin]
                center = np.mean(inner_quad1, axis=0)
                scale = 1.07  # inflation factor 7%
                roi_inner_quad2 = center + scale * (inner_quad2 - center)
                
                fid_x2, fid_y2, fid_r2 = detect_fiducial(roi2, roi_inner_quad2)
                if (fid_x2 or fid_y2 or fid_r2) is None:
                    print("Failed moving fiducial mark")
                    cv.imwrite(f"./analysis/moving_fid_failed/frame_{frame_num:05d}.png", outer_warped2)
                    mvn_fid_fail_cnt += 1
                else:
                    print("Detected fiducial mark")
                    fid_x2 += margin
                    fid_y2 += margin
                    
                    try:
                        # KEEP orientation normalization ONLY for visualization / saving
                        outer_warped_rot1, inner_quad_rot1, fid_rot1, rot_k1 = set_frame_orientation(outer_warped1, inner_quad1, (fid_x1, fid_y1), H_SIZE)
                        outer_warped_rot2, inner_quad_rot2, fid_rot2, rot_k2 = set_frame_orientation(outer_warped2, inner_quad2, (fid_x2, fid_y2), H_SIZE)
                    except:
                        print("FAILED FIDUCIAL ROTATION FUNCTION")
                        # Visualization
                        vis_frame1 = outer_warped1.copy()
                        cv.polylines(vis_frame1, [inner_quad1.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame1, (fid_x1, fid_y1), fid_r1, (0,255,0), 2)
                        vis_frame2 = outer_warped2.copy()
                        cv.polylines(vis_frame2, [inner_quad2.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame2, (fid_x2, fid_y2), fid_r2, (0,255,0), 2)
                        cv.imshow(f"static frame fiducial failed", vis_frame1)
                        cv.imshow(f"moving frame fiducial failed", vis_frame2)
                        keyp = cv.waitKey(0)
                    
                    print(f"FID1: {fid_rot1.reshape(1, 2)}")
                    print(f"FID2: {fid_x2}, {fid_y2}; ROT_K: {rot_k2}")
                    
                    if rot_k2 != 0:
                    #     # shape (4,2), shape (1,2)
                        src_pts = np.vstack([inner_quad2, (fid_x2, fid_y2)])
                        dst_pts = np.vstack([inner_quad_rot1, fid_rot1.reshape(1, 2)])
                        #src_pts, dst_pts, = inner_quad_rot2, inner_quad_rot1
                        H_by_fid, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)
                        outer_warped_rot2 = cv.warpPerspective(outer_warped2, H_by_fid, (H_SIZE, H_SIZE))
                    if H_by_fid is None:
                        print("Failed homography with fid")
                        cv.imwrite(f"./analysis/homography_fid_failed/frame_{frame_num:05d}.png", roi2)
                        hom_fid_fail_cnt += 1
                        
                        vis_frame1 = outer_warped_rot1.copy()
                        cv.polylines(vis_frame1, [inner_quad_rot1.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame1, (int(fid_rot1[0]), int(fid_rot1[1])), fid_r1, (0,0,255), 2)
                        cv.circle(vis_frame1, (fid_x1, fid_y1), fid_r1, (0,255,0), 2)
                        vis_frame2 = outer_warped_rot2.copy()
                        cv.polylines(vis_frame2, [inner_quad_rot2.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame2, (int(fid_rot2[0]), int(fid_rot2[1])), fid_r1, (0,0,255), 2)
                        cv.circle(vis_frame2, (fid_x2, fid_y2), fid_r2, (0,255,0), 2)
                        cv.imshow(f"Homography failed1", vis_frame1)
                        cv.imshow(f"Homography failed2", vis_frame2)
                        cv.waitKey(0)
                    else:
                        # Visualization
                        vis_frame1 = outer_warped_rot1.copy()
                        cv.polylines(vis_frame1, [inner_quad_rot1.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame1, (int(fid_rot1[0]), int(fid_rot1[1])), fid_r1, (0,0,255), 2)
                        cv.circle(vis_frame1, (fid_x1, fid_y1), fid_r1, (0,255,0), 2)
                        vis_frame2 = outer_warped_rot2.copy()
                        cv.polylines(vis_frame2, [inner_quad_rot2.astype(int)], True, (0,0,255), 2)
                        cv.circle(vis_frame2, (int(fid_rot2[0]), int(fid_rot2[1])), fid_r1, (0,0,255), 2)
                        cv.circle(vis_frame2, (fid_x2, fid_y2), fid_r2, (0,255,0), 2)
                        cv.imshow(f"static frame fiducial rotated", vis_frame1)
                        cv.imshow(f"moving frame fiducial rotated", vis_frame2)
                        keyp = cv.waitKey(5)
                        
                        # For saving the static camera frame, crop inner square from outer_warped
                        xs = inner_quad_rot1[:, 0]
                        ys = inner_quad_rot1[:, 1]

                        x0 = int(np.min(xs))
                        x1 = int(np.max(xs))
                        y0 = int(np.min(ys))
                        y1 = int(np.max(ys))

                        # safety clamp
                        x0 = max(0, x0)
                        y0 = max(0, y0)
                        x1 = min(outer_warped_rot1.shape[1], x1)
                        y1 = min(outer_warped_rot1.shape[0], y1)

                        saving_img = outer_warped_rot1[y0:y1, x0:x1]

                        if saving_img.size == 0:
                            print("INVALID INNER SQUARE SIZE")
                        else:
                            # Resize to canonical size
                            saving_img = cv.resize(saving_img, (INNER_SIZE, INNER_SIZE), interpolation=cv.INTER_AREA)
                            
                            # Visualization
                            cv.imshow(f"saving static image", saving_img)
                            keyp = cv.waitKey(1)
                            
                            # Convert to YUV
                            yuv = cv.cvtColor(saving_img, cv.COLOR_BGR2YUV)
                            Y, U, V = cv.split(yuv)
                            # Normalization
                            Y = Y.astype(np.float32) / 255.0    # l(x,y)
                            U = U.astype(np.float32) / 255.0
                            V = V.astype(np.float32) / 255.0
                            
                            # Light direction from moving camera
                            # r1, r2, r2 and t give camera orientation and position relative to the plane.
                            # H_norm = np.linalg.inv(K2) @ (np.linalg.inv(H_by_fid) @ H_outer2)    # [r1,r2,t] = K^-1 • K[r1,r2,t]
                            if rot_k2 == 0:
                                H_norm = np.linalg.inv(K2) @ H_outer2
                            else:
                                H_norm = np.linalg.inv(K2) @ (H_by_fid @ H_outer2)
                            
                            # R = H_norm[:,:2]
                            # T = H_norm[:,2]
                            # l = -R.T @ T
                            # u, v = l / np.linalg.norm(l)
                            # print(f"U: {u}, V: {v}, w: {np.sqrt(u**2+v**2)}")
                            #cv.waitKey(0)
                            
                            r1 = H_norm[:,0]
                            r2 = H_norm[:,1]
                            t = H_norm[:,2]
                            r1 /= np.linalg.norm(r1)
                            r2 /= np.linalg.norm(r2)
                            r3 = np.cross(r1, r2)
                            R_h = np.stack([r1, r2, r3], axis=1)  # This rotation maps plane coords -> camera coords
                            
                            # This vector points from the plane origin toward the camera
                            l_plane = -R_h.T @ t
                            l_norm = l_plane / np.linalg.norm(l_plane)
                            
                            # l_x, l_y, l_z
                            u, v, w = l_norm
                            # enforce the coords to be above surface
                            if w < 0: u, v, w = -u, -v, -w
                            print(f"U: {u}, V: {v}, w: {w}, SQRT(u*u+v*v): {np.sqrt(u*u+v*v)}")
                            
                            if u*u+v*v <= 1:
                                # Save
                                all_lights.append([u, v])
                                all_images.append(Y)
                                all_U.append(U)
                                all_V.append(V)
                                
                                # live plot
                                lights_np = np.array(all_lights)
                                sc.set_offsets(lights_np)
                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                    
        if save_path is not None:
            cv.imwrite(f"{save_path}/frame_{frame_num:05d}.png", vis_frame1)

        if SKIP_FRMS + frame_num >= video_length1: break
        if SKIP_FRMS + frame_num >= video_length2: break

        for _ in range(SKIP_FRMS): analysis_vid1.read()
        for _ in range(SKIP_FRMS): analysis_vid2.read()
        
        frame_num += (SKIP_FRMS + 1)
        frame_cnt += 1
        
        keyp = cv.waitKey(1)
        running = keyp != 113

    print(f"MOVING SQUARE DETECTION FAIL COUNT: {mvn_sqr_fail_cnt}")
    print(f"MOVING FIDUCIAL DETECTION FAIL COUNT: {mvn_fid_fail_cnt}")
    
    lights = np.array(all_lights, np.float32)           # (N, 2)
    images = np.stack(all_images, axis=0)               # (N, H, W)
    U_avg = np.mean(np.stack(all_U, axis=0), axis=0)    # (H, W)
    V_avg = np.mean(np.stack(all_V, axis=0), axis=0)    # (H, W)

    np.save("analysis/lights.npy", lights)
    np.save("analysis/images.npy", images)
    np.save("analysis/U_avg.npy", U_avg)
    np.save("analysis/V_avg.npy", V_avg)
    
    cv.destroyAllWindows()
    analysis_vid1.release()
    analysis_vid2.release()
    
    r = np.sqrt(lights[:,0]**2 + lights[:,1]**2)
    print("max radius:", r.max())
    print("min radius:", r.min())
    
    plt.ioff()
    plt.scatter(lights[:,0], lights[:,1])
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
