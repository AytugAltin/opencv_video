"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomText = (50,500)
fontScale              = 0.75
fontColor              = (255,255,255)
RedColor               = (255,0,0)
GreenColor             = (1,255,1)
BlueColor              = (0,0,255)
lineType               = 2
topLeftCornerOfText = (50,50)

start = 0


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool: 
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def skip(cap):
    if  between(cap, 0, start):
        return False
    return True
    

def write(frame,text,location,color):
    cv2.putText(frame,text, 
    location, 
    font, 
    fontScale,
    color,
    lineType)
    
def sub(frame,text,color = fontColor):
    write(frame,text,bottomText,color)
    
def info(frame,text,color = fontColor):
    write(frame,text,topLeftCornerOfText,color)
    
def grayscale(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sub(frame,'GrayScale')
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    return frame


def grayscale_zone(frame,cap):
    if  between(cap, 1000, 1500) or between(cap, 2000, 2500)  or between(cap, 3000, 3500):
        frame = grayscale(frame)
    return frame

def Gblur(frame,kernel,size):
    frame = cv2.GaussianBlur(frame,kernel, size)
    text = "GaussianBlur Kernelsize:" + (str(kernel))
    sub(frame,text)
    return frame

def smoothing_zone(frame,cap):
    if  between(cap, 5500, 6000) :
        frame = Gblur(frame,(5,5), 0)
        
    if  between(cap, 6000, 6500) :
        frame = Gblur(frame,(1,51), 0)
        
    if  between(cap, 6500, 7000) :
        frame = Gblur(frame,(51,1), 0)
        
    if  between(cap, 7000, 7500) :
        frame = Gblur(frame,(51,51), 0)
        
    if  between(cap, 7500, 8000) :
        frame = Gblur(frame,(101,101),0)
    
    return frame

def Biblur(frame,d,sigmaColor,sigmaSpace):
    frame = cv2.bilateralFilter(frame,d,sigmaColor,sigmaSpace)
    text = "Bi-LateralFilter  Diameter:" + str(d) + " sigmaColor:" + str(sigmaColor)+ " sigmaSpace:" + (str(sigmaSpace)) 
    sub(frame,text)
    return frame


def biLateral_zone(frame,cap):
    if  between(cap, 8000, 8500) :
        frame = Biblur(frame,9,75,75)
        
    if  between(cap, 8500, 9000) :
        frame = Biblur(frame,50,75,75)
        
    if  between(cap, 9000, 9500) :
        frame = Biblur(frame,9,500,75)
        
    if  between(cap, 9500, 10000) :
        frame = Biblur(frame,9,75,500)
        
    if  between(cap, 10000, 11000) :
        frame = Biblur(frame,9,1500,1500)
        
    
    return frame

def objectGrabbing_zone(frame,cap):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    text = ""
    if  between(cap, 13000, 19000):
        lower_bounds = np.array([26,82,119])
        upper_bounds = np.array([65,255,255])
        frame = cv2.inRange(frame,lower_bounds,upper_bounds)
        if  between(cap, 13000, 14000):
            frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        text = "Threshold color of ball"
        
        
    
    if  between(cap, 14000, 19000):
        kernel = np.ones((25,25),np.uint8)
        frame = cv2.dilate(frame,kernel,  iterations = 1)
        frame = cv2.erode(frame, kernel, iterations = 1)
        
        height  = frame.shape[0]
        width = frame.shape[1]
        
        left = np.zeros((height, width), np.uint8)
        
        frame = cv2.merge([np.zeros((height, width), np.uint8), 
                           frame,
                           np.zeros((height, width), np.uint8)])
        text = "Close ball(filling holes): Dilate + erode with kernel(25,25)"
        
    
    
    sub(frame,text)
    info(frame,"VHS space")
        
    
    return frame
    

def sobel_zone(frame,cap):
    frame2 = frame
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.GaussianBlur(frame2,(3,3),0)
    
    if  between(cap, 22500, 24000) :
        frame2 = cv2.Sobel(frame2,cv2.CV_8U,0,1,ksize=5)
        subtext = "Horizontal Edges Kernel size 5"
        
    if  between(cap, 24000, 26500) :
        frame2 = cv2.Sobel(frame2,cv2.CV_8U,0,1,ksize=1)
        subtext = "Horizontal Edges Kernel size 1"
    
    if  between(cap, 26500, 28000) :
        frame2 = cv2.Sobel(frame2,cv2.CV_8U,1,0,ksize=1)
        subtext = "Vertical Edges  Kernel size 1"
    
    purple = np.array([[[s,0,s] for s in r] for r in frame2],dtype="u1")
    
    tmp = cv2.cvtColor(purple, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(purple)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    purple_rgba = dst
    
    b_channel, g_channel, r_channel = cv2.split(frame)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    frame3 = cv2.add(img_BGRA,purple_rgba)
    frame3 = cv2.add(frame3,purple_rgba)
    frame3 = cv2.add(frame3,purple_rgba)
    frame3 = cv2.add(frame3,purple_rgba)
    frame3 = cv2.add(frame3,purple_rgba)
    
    sub(frame3,subtext)
    info(frame3, "Sobel Edge Detection")
    frame3 = cv2.cvtColor(frame3,cv2.COLOR_RGBA2RGB)
    return frame3
    

def hough_zone(frame,cap):
    # Code partially from Opencv Docs 
    
    if  between(cap, 28000, 30000):
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame2 = cv2.medianBlur(frame2, 9)
        rows = frame2.shape[0]
        circles = cv2.HoughCircles(frame2,cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2= 30, minRadius=1, maxRadius=50)
        color = (255, 0, 255)
        text = "Minradius=1 Maxradius=50 param2=30"
        
    if  between(cap, 30000, 32000):
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame2 = cv2.medianBlur(frame2, 9)
        rows = frame2.shape[0]
        circles = cv2.HoughCircles(frame2,cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2= 20, minRadius=1, maxRadius=50)
        color = (0, 0, 255)
        text = "Improvement1: Lowering param2 threshold"
        
    if  between(cap, 32000, 38000):
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame2 = cv2.medianBlur(frame2,9)
        rows = frame2.shape[0]
        circles = cv2.HoughCircles(frame2,cv2.HOUGH_GRADIENT, 1, rows/8, param1=100, param2=20, minRadius=10, maxRadius=23)
        color = (255, 0, 0)
        text = "Improvement2: MaxRadius from 50 to 25 + MinRadius from 1 to 10"
    

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(frame, center, 1, (0, 100, 100), 3) # circle center
            radius = i[2]
            cv2.circle(frame, center, radius, color, 3) # circle outline

    info(frame,"Hough Transform")
    sub(frame,text)
    
    return frame
    
    
    
def objectRe_zone(frame,cap):
    res = frame
    info(frame,"Template matching with Cross correlation (normed)")
    if  between(cap, 39000, 45000):
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        template = cv2.imread('bal_feature.png',0)
        w, h = template.shape[::-1]
        
        method =  cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(img,template,method )
        
        if  between(cap, 42000, 45000):
            
            scaled = (res + 1)*255/2
            res = scaled.astype(np.uint8)
            
            res = cv2.resize(res,(img.shape[1],img.shape[0]))
            res = cv2.cvtColor(res,cv2.COLOR_GRAY2RGB)
            info(res,"Likkelihood map Cross correlation (normed)")
            
            return res
        
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame,top_left, bottom_right, 255, 2)
        
    
    return frame
    

def replace(frame):
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('finale_moment.png',0)
    
    w, h = template.shape[::-1]
    
    
    method =  cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(img,template,method )
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame,top_left, bottom_right, 255, 2)
        
    img  = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    img_overlay_rgba = cv2.imread("fireball.png",-1)
    x = int(img_overlay_rgba.shape[0]/4)
    y = int(img_overlay_rgba.shape[1]/4)
    img_overlay_rgba = cv2.resize(img_overlay_rgba,(x,y) )
    
    x = int(img_overlay_rgba.shape[0])
    y = int(img_overlay_rgba.shape[1])
    
    x_image =  int((bottom_right[0] + top_left[0])/2 - x/2)
    y_image =  int((bottom_right[1] + top_left[1])/2 - y/2)
    
    alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
    img_result = img[:, :, :3].copy()
    img_overlay = img_overlay_rgba[:, :, :3]
    overlay_image_alpha(img_result, img_overlay, x_image, y_image, alpha_mask)
    
    return img_result


def carte_blanche(frame,cap):
    if  between(cap, 46000, 50000):
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        #save the image(i) in the same directory
        img = frame
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            frame = img
            sub(frame,"Face Detection")
            
    if  between(cap, 51000, 53000):
        frame = replace(frame)
        sub(frame,"Object replacement")
        
    if  between(cap, 54000, 55000) or between(cap, 56000, 57000) or between(cap, 58000, 59000):
        frame = cv2.flip(frame, 1)
        sub(frame,"Flip")
            
    return frame
            

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Code from https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv"""
    
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    

def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if skip(cap): #TODO remove line
                
                if cv2.waitKey(28) & 0xFF == ord('q'):
                    break
                
                if frame is None:
                    break
            
                
                if  between(cap, 1000, 3500):
                    frame = grayscale_zone(frame,cap)
                    
                    
                if  between(cap, 5000, 8000):
                    frame = smoothing_zone(frame,cap)
                 
                if  between(cap, 8000, 11000):
                    frame = biLateral_zone(frame,cap)
                    
                if  between(cap, 11000, 19000):
                    frame = objectGrabbing_zone(frame,cap)
                    
                if  between(cap, 22500, 28000):
                    frame = sobel_zone(frame,cap)
                    
                if  between(cap, 28000, 38000):
                    frame = hough_zone(frame,cap)
                    
                if  between(cap, 39000, 45000):
                    frame = objectRe_zone(frame,cap)
                    
                if  between(cap, 45000, 60000):
                    frame = carte_blanche(frame,cap)
                 
                 
                # (optional) display the resulting frame
                cv2.imshow('Frame', frame)
    
                # write frame that you processed to output
                #print(frame.shape)
                out.write(frame)
                
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
