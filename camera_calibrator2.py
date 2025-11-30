#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import json

COLS = 9
ROWS = 6
SKIP_FRMS = 40

def calibrate(video):
    cal_vid = cv.VideoCapture(video)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    # (X, Y, Z) => the coordinates of checker board!! The ideal 3D location of each corner
    # assumption: each corner has Z=0
    objp = np.zeros((ROWS*COLS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:COLS,0:ROWS].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    frame_count = 0
    success = 1
    video_length = int(cal_vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    while success:
        frame_count += 1
        success, frame = cal_vid.read()
        
        if success:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            ret, corners = cv.findChessboardCorners(gray, (COLS, ROWS), None)
            
            # if found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                cv.drawChessboardCorners(frame, (COLS, ROWS), corners2, ret)
                cv.imshow('frame', frame)
                cv.waitKey(1)
            
            if SKIP_FRMS * frame_count >= video_length: break
            
            for _ in range(SKIP_FRMS): cal_vid.read()
        
    cv.destroyAllWindows()
    cal_vid.release()
    
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
def show_and_save(ret, K, dist, rvecs, tvecs, path):
    print(f"Camera matrix: {K}\n")
    print(f"Distortion coefficient: {dist}\n")
    print(f"Rotation Vectors: {rvecs}\n")
    print(f"Translation Vectors: {tvecs}\n")

    with open(path, "w") as f:
        json.dump({"K": K.tolist(), "distortion": dist.tolist(), "rms": float(ret)}, f, indent=4)
    
    print("Intrinsics saved successfully")
    
def main():
    print("Start Camera Calibration")
    
    print("Static Camera:")
    ret, K, dist, rvecs, tvecs = calibrate("./data/static/calibration.MOV")
    show_and_save(ret, K, dist, rvecs, tvecs, "./calib/static/intrinsics.json")
    
    print("Moving Camera:")
    ret, K, dist, rvecs, tvecs = calibrate("./data/moving/calibration.MOV")
    show_and_save(ret, K, dist, rvecs, tvecs, "./calib/moving/intrinsics.json")
    
    print("End Camera Calibration")

if __name__ == "__main__":
    main()