from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import argparse
import numpy as np
from opencv_process_video import *
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
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
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture("bal.mp4")
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

def run():  
    on_low_H_thresh_trackbar(26)
    on_high_H_thresh_trackbar(70)
    on_low_S_thresh_trackbar(82)
    on_high_S_thresh_trackbar(251)
    on_low_V_thresh_trackbar(179)
    on_high_V_thresh_trackbar(255)
    i = 4
    while True:
        cap = cv.VideoCapture("balk.mp4")
        while True:
            
            ret, frame = cap.read()
            #frame = cv.imread("bal3.png")
            if frame is None:
                break
            
            #frame = Gblur(frame,(15,15), 0)
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            #frame_threshold = cv.inRange(frame_HSV, (6,157,56), (12,242,242))
            
            
            kernel = np.ones((5,5),np.uint8)
            
            
            kernelc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(i,i))
            
            frame_threshold = cv2.erode(frame_threshold, kernelc, iterations = 1) 
            frame_threshold = cv2.dilate(frame_threshold,kernelc,  iterations = 1)
            #frame_threshold = cv2.add(frame_threshold , cv2.Canny(frame_threshold,400,500))
            #frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 1)
            
            #frame_threshold= cv2.morphologyEx(frame_threshold,cv2.MORPH_OPEN,kernel, iterations = 2)
            
            
            cv.imshow(window_capture_name, frame)
            cv.imshow(window_detection_name, frame_threshold)
            
            
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                return 
            
            key = cv.waitKey(30)
            if key == ord('i') or key == 20:
                i = i + 1
                print(i)

            
x = run()