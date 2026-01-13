import sys
import math
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

SKIP_FRMS = 0
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
    dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
    
    # Find teh projective transform that maps the detected square in the image to a perfectly aligned square of the size.
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
    
    print(f"RADIUS SIZE: {r}")
    # Visualization
    # vis_frame1 = masked.copy()
    # cv.circle(vis_frame1, (int(xc), int(yc)), int(r), (0,255,0), 2)
    # cv.imshow(f"frame fiducial", vis_frame1)
    # cv.waitKey(1)
    
    if r < 5 or r > 80: return None, None, None
    
    return int(xc), int(yc), int(r)
    
def set_frame_orientation(warped, fiducial_pos, size=H_SIZE):
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
    
    # static cam detection only requires once and then all the frames share the same detections (outer, inner, fiducial)
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
        # vis_frame1 = frame1.copy()
        # cv.polylines(vis_frame1, [outer_quad1.astype(int)], True, (0,0,255), 2)
        # cv.imshow(f"static_frame outer", vis_frame1)
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
        # cv.imshow(f"static_frame inner", vis_frame1)
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
        # cv.imshow(f"static_frame fiducial", vis_frame1)
        # cv.waitKey(1)
        
        if (outer_quad1 is not None and inner_quad1 is not None) and (fid_x1 is not None and fid_y1 is not None and fid_r1 is not None): break
        
        if save_path is not None:
            cv.imwrite(f"{save_path}/frame_{frame_num:05d}.png", vis_frame1)

        if 1 + frame_num >= video_length1: break
    
    
    video_length1 = int(analysis_vid1.get(cv.CAP_PROP_FRAME_COUNT))
    video_length2 = int(analysis_vid2.get(cv.CAP_PROP_FRAME_COUNT))
    inner_quad2 = inner_quad1
    running = True
    frame_count = 0
    frame_num = 1
    all_images = [] # from static cam
    all_lights = [] # from moving cam
    all_U = []  # for colorizing
    all_V = []
    mvn_sqr_fail_cnt = 0
    mvn_fid_fail_cnt = 0
    
    while running:
        keyp = cv.waitKey(1)
        running = keyp != 113  # Press q to exit
        frame_count += 1
        success1, frame1 = analysis_vid1.read()
        success2, frame2 = analysis_vid2.read()
        if not (success1 and success2): break

        frame1 = cv.undistort(frame1.copy(), K1, dist1)
        frame2 = cv.undistort(frame2.copy(), K2, dist2)

        margin = int(0.01 * H_SIZE)   # crop 1% of width due to detection of outer square!
        
        # static cam uses the same quad for consistency!
        frame1[margin:-margin, margin:-margin]
        dst_pts = np.array([[0,0], [H_SIZE,0], [H_SIZE,H_SIZE], [0,H_SIZE]]) 
        H_outer1, _ = cv.findHomography(outer_quad1, dst_pts)
        if H_outer1 is not None:
            outer_warped1 = cv.warpPerspective(frame1, H_outer1, (H_SIZE, H_SIZE))
            
            # of course moving cam keeps to find new one
            roi2 = frame2[margin:-margin, margin:-margin]
            outer_quad2, outer_warped2, H_outer2, _ = detect_square(roi2)
            if outer_quad2 is None or H_outer2 is None:
                print("Failed moving outer square")
                cv.imwrite(f"./analysis/moving_sqr_failed/frame_{frame_num:05d}.png", roi2)
                mvn_sqr_fail_cnt += 1
            else:
                print("Detected moving outer square")
                
                # Visualization
                vis_frame1 = frame1.copy()
                cv.polylines(vis_frame1, [outer_quad1.astype(int)], True, (0,0,255), 2)
                vis_frame2 = frame2.copy()
                cv.polylines(vis_frame2, [outer_quad2.astype(int)], True, (0,0,255), 2)
                cv.imshow(f"static_frame outer", vis_frame1)
                cv.imshow(f"moving_frame outer", vis_frame2)
                cv.waitKey(1)
                
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
                cv.imshow(f"moving_frame inner", vis_frame2)
                cv.waitKey(1)
                
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
                    # fid_x2 += margin
                    # fid_y2 += margin
                    
                    # Rotation by fiducial mark position
                    outer_warped1, fid_rot1, R_f1 = set_frame_orientation(outer_warped1, (fid_x1, fid_y1), H_SIZE)
                    outer_warped2, fid_rot2, R_f2 = set_frame_orientation(outer_warped2, (fid_x2, fid_y2), H_SIZE)
                    
                    # Visualization
                    vis_frame1 = outer_warped1.copy()
                    cv.polylines(vis_frame1, [inner_quad1.astype(int)], True, (0,0,255), 2)
                    cv.circle(vis_frame1, (fid_x1, fid_y1), fid_r1, (0,255,0), 2)
                    vis_frame2 = outer_warped2.copy()
                    cv.polylines(vis_frame2, [inner_quad2.astype(int)], True, (0,0,255), 2)
                    cv.circle(vis_frame2, (fid_x2, fid_y2), fid_r2, (0,255,0), 2)
                    cv.imshow(f"static_frame fiducial", vis_frame1)
                    cv.imshow(f"moving_frame fiducial", vis_frame2)
                    cv.waitKey(1)
                    
                    # For saving the static camera frame, crop inner square from outer_warped
                    xs = inner_quad1[:, 0]
                    ys = inner_quad1[:, 1]

                    x0 = int(np.min(xs))
                    x1 = int(np.max(xs))
                    y0 = int(np.min(ys))
                    y1 = int(np.max(ys))

                    # safety clamp
                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    x1 = min(outer_warped1.shape[1], x1)
                    y1 = min(outer_warped1.shape[0], y1)

                    saving_img = outer_warped1[y0:y1, x0:x1]

                    if saving_img.size == 0:
                        print("INVALID INNER SQUARE SIZE")
                    else:
                        # Resize to canonical size
                        saving_img = cv.resize(saving_img, (INNER_SIZE, INNER_SIZE), interpolation=cv.INTER_AREA)
                        
                        # Visualization
                        cv.imshow(f"saving static image", saving_img)
                        cv.waitKey(1)
                        
                        # Convert to YUV
                        yuv = cv.cvtColor(saving_img, cv.COLOR_BGR2YUV)
                        Y, U, V = cv.split(yuv)
                        # Normalization
                        Y = Y.astype(np.float32) / 255.0    # l(x,y)
                        U = U.astype(np.float32) / 255.0
                        V = V.astype(np.float32) / 255.0
                        
                        # Now to get the light direction from moving camera
                        H_mat = H_outer2                      # H = K[r1,r2,t]
                        H_norm = np.linalg.inv(K2) @ H_mat    # [r1,r2,t]
                        r1 = H_norm[:,0]
                        r2 = H_norm[:,1]
                        r1 /= np.linalg.norm(r1)
                        r2 /= np.linalg.norm(r2)
                        r3 = np.cross(r1, r2)
                        R_h = np.stack([r1, r2, r3], axis=1)  # This rotation maps plane coords -> camera coords

                        t = H_norm[:, 2]
                        cam_pos_plane = -R_h.T @ t                # This vector points from the plane origin toward the camera
                        l = cam_pos_plane / np.linalg.norm(cam_pos_plane)   # [u v w]

                        # Apply fiducial-corrected rotation from a warped frame
                        l_surface = R_f2.T @ l
                        
                        #print("Light direction (surface frame):", l_surface)
                        
                        u, v = l_surface[0], l_surface[1]

                        # validity check (w = sqrt(1 - u^2 - v^2))
                        if u*u + v*v > 1.0:
                            print(f"INVALID: u*u + v*v = {u*u + v*v} > 1.0")
                        else:
                            # Save
                            all_lights.append([u, v])
                            all_images.append(Y)
                            all_U.append(U)
                            all_V.append(V)
                    
        if save_path is not None:
            cv.imwrite(f"{save_path}/frame_{frame_num:05d}.png", vis_frame1)

        if SKIP_FRMS + frame_num >= video_length1: break
        if SKIP_FRMS + frame_num >= video_length2: break

        for _ in range(SKIP_FRMS): analysis_vid1.read()
        for _ in range(SKIP_FRMS): analysis_vid2.read()
        frame_num += (SKIP_FRMS + 1)
        print(f"FRAME NUMBER {frame_num:05d}")

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
    
    plt.scatter(lights[:,0], lights[:,1])
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()
