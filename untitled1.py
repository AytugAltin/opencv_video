from __future__ import print_function
import cv2 as cv
import cv2 as cv2
import argparse
import numpy as np
from opencv_process_video import *
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
    i = 1
    while True:
        cap = cv.VideoCapture("edges.mp4")
        while True:
            
            ret, frame = cap.read()
            #frame = cv.imread("bal3.png")
            if frame is None:
                break
            
        
            
            
            frame1 = frame
            cv.imshow("first", frame1)
            
            
            frame2 = frame
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.GaussianBlur(frame2,(3,3),0)
            frame2 = cv2.Sobel(frame2,cv2.CV_8U,0,1,ksize=i)
            cv.imshow("second", frame2)
            
            #frame2 = cv2.Sobel(frame2,cv2.CV_64F,0,i,ksize=5)
            
            #frame2 = cv2.Sobel(frame,cv2.CV_64F,0,i,ksize=5)
            
            
            
            #frame4 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frame4 = np.array([[[0,s,0] for s in r] for r in frame2],dtype="u1")
            
            
            tmp = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
            _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
            b, g, r = cv2.split(frame4)
            rgba = [b,g,r, alpha]
            dst = cv2.merge(rgba,4)
            green_rgba = dst
            cv.imshow("OG", green_rgba)
            
            
            b_channel, g_channel, r_channel = cv2.split(frame)

            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
            
            img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            
            frame3 = cv.addWeighted(img_BGRA,0.3,green_rgba,0.7,0)
            frame3 = cv.add(img_BGRA,green_rgba)
            frame3 = cv.add(frame3,green_rgba)
            frame3 = cv.add(frame3,green_rgba)
            frame3 = cv.add(frame3,green_rgba)
            frame3 = cv.add(frame3,green_rgba)
            cv.imshow("Third", frame3)
            
            
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                return 0
            
            key = cv.waitKey(30)
            if key == ord('i') or key == 20:
                i = i + 2
                print(i)

            
x = run()