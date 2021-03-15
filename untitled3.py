from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import argparse
import numpy as np
from opencv_process_video import *
from matplotlib import pyplot as plt



def run():  
    i = 4
    while True:
        cap = cv.VideoCapture("od.mp4")
        while True:
            
            ret, frame = cap.read()
            # frame = cv.imread("homeball.png")
            if frame is None:
                break
            
            
        
            #cv.imshow("name", frame)
            
            
            img = frame
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img2 = img.copy()
            template = cv.imread('bal_feature.png',0)
            
            w, h = template.shape[::-1]
            
            img = img2.copy()
            
            method =  cv.TM_CCOEFF
            res = cv.matchTemplate(img,template,method )
            
                
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            
            
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
                
            
         
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(frame,top_left, bottom_right, 255, 2)
            
            
            
            
            cv2.imshow('Frame', frame)
            cv2.imshow('res', res)
            
            img_not =  abs(res-1)
            cv2.imshow('res2', img_not)
            
            
                            
           
        
            
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                return 
            
            key = cv.waitKey(30)
            if key == ord('i') or key == 20:
                i = i + 1
                print(i)

            
x = run()