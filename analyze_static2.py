import sys
import cv2 as cv

SKIP_FRMS = 100

def detect_edges(video, save_path=None):
    analysis_vid = cv.VideoCapture(video)
    
    window_name = ('Sobel - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    
    frame_count = 0
    success = 1
    video_length = int(analysis_vid.get(cv.CAP_PROP_FRAME_COUNT))
    
    while success:
        frame_count += 1
        success, frame = analysis_vid.read()
        
        if success:
            # Load the image
            #src = cv.imread(, cv.IMREAD_COLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Check if image is loaded fine
            # if gray is None:
            #     print ("Error opening image")
            #     return -1
            
            # Reduce noise by blurring with a Gaussian filter (kernel size = 3)
            gray_blur = cv.GaussianBlur(gray, (3, 3), 0)
            # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            
            
            grad_x = cv.Sobel(gray_blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
            grad_y = cv.Sobel(gray_blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
            
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
            
            # convert back to CV_8U
            abs_grad_x = cv.convertScaleAbs(grad_x)
            abs_grad_y = cv.convertScaleAbs(grad_y)
            
            # try to approximate the gradient by adding both directional gradients
            grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            # note: not an exact calculation at all, but it is good for the purposes
            
            print(f"frame count: {frame_count}")
            
            # cv.imshow(window_name, grad)
            # cv.waitKey(1)
            # cv.destroyWindow(window_name)
            
            if save_path is not None:
                cv.imwrite(f"{save_path}/frame_{frame_count*SKIP_FRMS:05d}.png", grad)
            
            if SKIP_FRMS * frame_count >= video_length: break
            
            #analysis_vid.set(cv.CAP_PROP_POS_FRAMES, frame_count*SKIP_FRMS)
            for _ in range(SKIP_FRMS): analysis_vid.read()
            
        
    cv.destroyAllWindows()
    analysis_vid.release()
    
    return 000
    




def main():
    # detect_edges("./data/static/coin.MOV", "./data/static/sobel_imgs")
    # detect_edges("./data/moving/coin.MOV", "./data/moving/sobel_imgs")
    detect_edges("./data/static/coin.MOV")
    detect_edges("./data/moving/coin.MOV")
    

if __name__ == "__main__":
    main()