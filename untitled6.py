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
        cap = cv.VideoCapture("myface.mp4")
        while True:
            
            ret, frame = cap.read()
            # frame = cv.imread("homeball.png")
            if frame is None:
                break
            
            
        
            
            
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
            #save the image(i) in the same directory
            img = frame
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
            
            
            cv2.imshow('img',img)
            
            
            
            
            
            
            
            
                            
           
        
            
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                return 
            
            key = cv.waitKey(30)
            if key == ord('i') or key == 20:
                i = i + 1
                print(i)

            
x = run()