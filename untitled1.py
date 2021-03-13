from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import argparse
import numpy as np
from opencv_process_video import *
from matplotlib import pyplot as plt


max_value = 1000
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = 0
high_S = 0
high_V = 0
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    #low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    #high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    #low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    #high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    #low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    #high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
# cap = cv.VideoCapture("bal.mp4")
# cv.namedWindow(window_capture_name)
# cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)




def run():
    x = 1
    d = 2
    
    while True:
        cap = cv.VideoCapture("bal.mp4")
        while True:
            
            ret, frame = cap.read()
            
            if frame is None:
                break
            
            cv2.imshow('Frame', frame)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # img2 = img.copy()
            # template = cv.imread('bal_feature.png',0)
            
            # w, h = template.shape[::-1]
            # # All the 6 methods for comparison in a list
            # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
            
            # for meth in methods:
            #     img = img2.copy()
            #     method = eval(meth)
            #     # Apply template Matching
            #     res = cv.matchTemplate(img,template,method)
            #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            #         top_left = min_loc
            #     else:
            #         top_left = max_loc
            #     bottom_right = (top_left[0] + w, top_left[1] + h)
            #     cv.rectangle(img,top_left, bottom_right, 255, 2)
            #     # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            #     # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            #     # plt.subplot(122),plt.imshow(img,cmap = 'gray')
            #     # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                
            #     # plt.suptitle(meth)
            #     # plt.show()
                
            
            # cv2.imshow('Frame', frame)
                            
    

            
run()