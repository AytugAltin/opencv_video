# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:08:28 2021

@author: Tug
"""

import cv2 as cv
import cv2 as cv2
import numpy as np
from opencv_process_video import *
from matplotlib import pyplot as plt


def run():
    x = 1
    d = 2
    
    while True:
        cap = cv.VideoCapture("bal.mp4")
        while True:
            
            ret, frame = cap.read()
            
            if frame is None:
                break
            
            cv.imshow('Frame', frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img2 = img.copy()
            template = cv.imread('bal_feature.png',0)
            
            w, h = template.shape[::-1]
            # All the 6 methods for comparison in a list
            methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                        'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
            
            for meth in methods:
                img = img2.copy()
                method = eval(meth)
                # Apply template Matching
                res = cv.matchTemplate(img,template,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv.rectangle(img,top_left, bottom_right, 255, 2)
                # plt.subplot(121),plt.imshow(res,cmap = 'gray')
                # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                # plt.subplot(122),plt.imshow(img,cmap = 'gray')
                # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                
                # plt.suptitle(meth)
                # plt.show()
                
            
            cv2.imshow('Frame', frame)
                            
            
x= run()