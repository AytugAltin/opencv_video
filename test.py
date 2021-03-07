# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:19:13 2021

@author: Tug
"""

"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomText = (200,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
topLeftCornerOfText = (50,50)


# helper function to change what you do based on video seconds


def write(frame,text):
    cv2.putText(frame,text, 
    bottomText, 
    font, 
    fontScale,
    fontColor,
    lineType)

frame = cv2.imread('image.png',1)
i = 1

while 1 == 1:
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(25) & 0xFF == ord('i'):
        i=i+5
        print(i)
    
    frame2 = cv2.bilateralFilter(frame,i,75,75)
    write(frame2,str(i))
    cv2.imshow('image',frame2)
        

cv2.destroyAllWindows()
