import sys
import cv2 as cv
import numpy as np

SKIP_FRMS = 100
SCALE = 9   # kernel size
K = 0.04     # optimal: 0.04 – 0.15
THRESH = 0.01
NMS_WINSIZE = 13

def detect_edges(video, save_path=None):
    analysis_vid = cv.VideoCapture(video)
    
    window_name = ('Sobel - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    
    running = True
    image_to_show = 0
    frame_count = 0
    success = 1
    video_length = int(analysis_vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    while running and success:
        frame_count += 1
        success, frame = analysis_vid.read()
        
        if success:
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            
            # Sobel with float normalization (divide by 255.0)
            grad_x = cv.Sobel(gray, cv.CV_32F, dx=1, dy=0, ksize=3) / 255.0
            grad_y = cv.Sobel(gray, cv.CV_32F, dx=0, dy=1, ksize=3) / 255.0
            """
            Calculate the "derivatives" in x and y directions. Use the function cv.Sobel() as shown below:
                gray_blur: the input image (CV_8U)
                grad_x | grad_y: the output image
                ddepth: the depth of the output image (CV_16S to avoid overflow)
                x_order: the order of the derivative in x direction
                y_order: the order of the derivative in y direction
                scale, delta and borderType: default
            Notice that to calculate the gradient in x direction we use: x_order=1 and y_order=0,
            and do analogously for the y direction.
            From: https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
            """
            
            # True gradient magnitude (Euclidean norm of the gradient vectors) for correct edge strength!
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Compute the local averages of squared derivatives for each pixel
            Hxx = cv.GaussianBlur(grad_x * grad_x, (SCALE, SCALE), sigmaX = 0, sigmaY = 0)
            Hxy = cv.GaussianBlur(grad_x * grad_y, (SCALE, SCALE), sigmaX = 0, sigmaY = 0)
            Hyy = cv.GaussianBlur(grad_y * grad_y, (SCALE, SCALE), sigmaX = 0, sigmaY = 0)
            """
            Gaussian blur acts as the local window / integration kernel:
                corners become smoother and stabler but lose spatial precision.
            These three images form the structure tensor at each pixel:
                M = ((H_xx, H_xy), 
                     (H_xy, H_yy)).
            """

            # Standard Harris corner measure
            Hresp = (Hxx*Hyy - Hxy*Hxy) - K*(Hxx+Hyy)**2
            """
            R = det(M) - k•(trace(M))^2
            Interpretation:
                Large positive R: corner-like (both eigenvalues large);
                Negative R: edge-like (one eigenvalue large, other small);
                Near zero: flat region.
            Harris constant (K): controls how sensitive the detector is to corners vs. edges.
            """

            # Non-maxima suppression 
            Hrespd = cv.dilate(Hresp, cv.getStructuringElement(cv.MORPH_ELLIPSE, (NMS_WINSIZE, NMS_WINSIZE)))
            corners = ((Hresp>THRESH) * (Hresp == Hrespd)).astype(np.uint8)*255
            corners_loc = np.argwhere(corners > 0)
            """
            cv.dilate(Hresp, kernel) expands each bright spot into a local maximum region of size given.
            corners: combined with both threshold-ed Hresp and local maxima yielding a binary mask.
            corners_loc: the pixel coordinates of those detected corner points.
            """
            
            shown_img = [frame, grad_mag, Hresp][image_to_show]

            if shown_img.dtype != np.uint8:
                # Scale min-max to 0-1 if image is floating point
                cv.normalize( shown_img, shown_img, 0, 1, norm_type=cv.NORM_MINMAX )
            
            if shown_img.ndim<3 or shown_img.shape[2]==1:
                # Transform to color image so we can later plot coloured circles
                shown_img = cv.cvtColor(shown_img, cv.COLOR_GRAY2BGR)

            for corner in corners_loc:
                cv.circle(shown_img, (corner[1], corner[0]), 3, (0,0,255), 2, cv.LINE_AA)

            cv.imshow('frame', shown_img )
            
            if save_path is not None:
                cv.imwrite(f"{save_path}/frame_{frame_count*SKIP_FRMS:05d}.png", shown_img)
            
            if SKIP_FRMS * frame_count >= video_length: break
            
            for _ in range(SKIP_FRMS): analysis_vid.read()
            
            keyp = cv.waitKey(1)
            #cam.control(keyp)
            running = keyp != 113  # Press q to exit
            if keyp == 105: # Press i to change the displayed image
                image_to_show = (image_to_show+1)%3
            
        
    cv.destroyAllWindows()
    analysis_vid.release()
    
    return 000
    


def detect_edges2(video, save_path=None):
    analysis_vid = cv.VideoCapture(video)
    
    running = True
    image_to_show = 0
    frame_count = 0
    success = 1
    video_length = int(analysis_vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    while running and success:
        frame_count += 1
        success, frame = analysis_vid.read()
        
        if success:
            # gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            # grad_sharp = cv.GaussianBlur(gray, (5,5), 0)   # smoother background
            # grad_sharp = cv.addWeighted(gray, 1.5, grad_sharp, -0.5, 0)  # sharpen edges
            edges = cv.Canny(frame, 130, 200)
            
            
            contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.drawContours(edges, contours, -1, (0,0,255), 2)  # sort the detected contours to make out the square!!!
            
            cv.imshow('frame', edges)
            
            # if save_path is not None:
            #     cv.imwrite(f"{save_path}/frame_{frame_count*SKIP_FRMS:05d}.png", shown_img)
            
            if SKIP_FRMS * frame_count >= video_length: break
            
            for _ in range(SKIP_FRMS): analysis_vid.read()
            
            keyp = cv.waitKey(1)
            #cam.control(keyp)
            running = keyp != 113  # Press q to exit
            if keyp == 105: # Press i to change the displayed image
                image_to_show = (image_to_show+1)%3
            
        
    cv.destroyAllWindows()
    analysis_vid.release()
    
    return 000
            

def main():
    # detect_edges("./data/static/coin.MOV", "./data/static/sobel_imgs")
    # detect_edges("./data/moving/coin_shifted.MOV", "./data/moving/sobel_imgs")
    detect_edges2("./data/static/coin.MOV")
    #detect_edges("./data/static/coin.MOV")
    #detect_edges2("./data/moving/coin_shifted.MOV")
    

if __name__ == "__main__":
    main()