import argparse
import cv2
import sys
import numpy as np
from opencv_process_video import *


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()



def run():
    while True:
        cap = cv2.VideoCapture("bal2.mp4")
        while True:
            
            ret, frame = cap.read()
            if frame is None:
                break
            
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_bounds = np.array([5,99,59])
            upper_bounds = np.array([17,225,253])
            frame = cv2.inRange(hsv,lower_bounds,upper_bounds)
            
            
            
            frame = Biblur(frame,9,75,75)
            
            kernel = np.ones((5,5),np.uint8)
            frame = cv2.dilate(frame,kernel,iterations = 1)
            frame = cv2.erode(frame, kernel, iterations=1) 
            
            # if  between(cap, 1000, 1500) or between(cap, 2000, 2500)  or between(cap, 3000, 3500):
            #     frame = cv2.add(frame , cv2.Canny(frame,200,200))
            #     frame = cv2.dilate(frame,kernel,iterations = 1)
            #     frame = cv2.erode(frame, kernel, iterations=1) 
            
            #frame = cv2.dilate(frame,kernel,iterations = 1)
            
            
                
            
            cv2.imshow("FRAME", frame)
            
            key = cv2.waitKey(30)
            if key == ord('q') or key == 27:
                return 0
            
            
x = run()
