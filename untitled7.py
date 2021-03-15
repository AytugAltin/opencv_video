from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import argparse
import numpy as np
from opencv_process_video import *
from matplotlib import pyplot as plt



def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err

def run():  
    i = 4
    while True:
        cap = cv.VideoCapture("finale.mp4")
        while True:
            
            ret, frame = cap.read()
            # frame = cv.imread("homeball.png")
            if frame is None:
                break
            
            
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            template = cv.imread('finale_moment.png',0)
            
            w, h = template.shape[::-1]
            
            
            method =  cv.TM_SQDIFF_NORMED
            res = cv.matchTemplate(img,template,method )
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
        
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
                
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(frame,top_left, bottom_right, 255, 2)
                
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
                        
            
                        
            cv2.imshow('img',img_result)
            
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                return 
            
            key = cv.waitKey(30)
            if key == ord('i') or key == 20:
                i = i + 1
                print(i)
                
        

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
            
x = run()