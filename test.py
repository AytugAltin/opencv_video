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

import numpy as np
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

target = cv2.imread('bal.png',1)


mask = np.zeros_like(target)
mask = cv2.circle(mask, (25,25), 25, (255,255,255), -1)
cv2.imshow("Search Region" , mask)

cv2.waitKey(0)

SearchImage = cv2.bitwise_and(target,target,mask = mask)

cv2.imshow("Search Region" , SearchImage)
cv2.waitKey()

#convert RGBto Lab
LabImage = cv2.cvtColor(SearchImage,cv2.COLOR_BGR2LAB)

cv2.imshow("Lab(b)" , LabImage[:, :, 1])
cv2.waitKey()

ret,Binary = cv2.threshold(LabImage[:, :, 1], 0, 255, cv2.THRESH_OTSU)
cv2.imshow('win1', Binary)
cv2.waitKey(0)

 #find contours
contours, hierarchy = cv2.findContours(Binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(target.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

for cnt in contours:

   x, y, w, h = cv2.boundingRect(cnt)
   aspect_ratio = float(w) / h

   area = cv2.contourArea(cnt)
   x, y, w, h = cv2.boundingRect(cnt)
   rect_area = w * h
   extent = float(area) / rect_area

   hull = cv2.convexHull(cnt)
   hull_area = cv2.contourArea(hull)
   solidity = float(area) / hull_area

   equi_diameter = np.sqrt(4 * area / np.pi)

   (x, y), (MA, ma), Orientation = cv2.fitEllipse(cnt)

   print(" Width = {}  Height = {} area = {}  aspect ration = {}  extent  = {}  solidity = {}   equi_diameter = {}   orientation = {}"
         .format(  w , h , area ,   aspect_ratio , extent , solidity , equi_diameter , Orientation))



cv2.imshow('win1', img_contours)
cv2.waitKey(0)
