"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomText = (200,500)
fontScale              = 0.75
fontColor              = (255,255,255)
lineType               = 2
topLeftCornerOfText = (50,50)

start = 14000


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool: 
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def skip(cap):
    if  between(cap, 0, start):
        return False
    return True
    


def write(frame,text,location):
    cv2.putText(frame,text, 
    location, 
    font, 
    fontScale,
    fontColor,
    lineType)
    
def sub(frame,text):
    write(frame,text,bottomText)
    
def info(frame,text):
    write(frame,text,topLeftCornerOfText)
    
def grayscale(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sub(frame,'GrayScale')
    return frame


def grayscale_zone(frame,cap):
    if  between(cap, 1000, 1500) or between(cap, 2000, 2500)  or between(cap, 3000, 3500):
        frame = grayscale(frame)
    return frame

def Gblur(frame,kernel,size):
    frame = cv2.GaussianBlur(frame,kernel, size)
    text = "GaussianBlur Kernelsize:" + (str(kernel))
    sub(frame,text)
    return frame

def smoothing_zone(frame,cap):
    if  between(cap, 5500, 6000) :
        frame = Gblur(frame,(5,5), 0)
        
    if  between(cap, 6000, 6500) :
        frame = Gblur(frame,(1,51), 0)
        
    if  between(cap, 6500, 7000) :
        frame = Gblur(frame,(51,1), 0)
        
    if  between(cap, 7000, 7500) :
        frame = Gblur(frame,(51,51), 0)
        
    if  between(cap, 7500, 8000) :
        frame = Gblur(frame,(101,101),0)
    
    return frame

def Biblur(frame,d,sigmaColor,sigmaSpace):
    frame = cv2.bilateralFilter(frame,d,sigmaColor,sigmaSpace)
    text = "Bi-LateralFilter  Diameter:" + str(d) + " sigmaColor:" + str(sigmaColor)+ " sigmaSpace:" + (str(sigmaSpace)) 
    sub(frame,text)
    return frame


def biLateral_zone(frame,cap):
    if  between(cap, 8000, 8500) :
        frame = Biblur(frame,9,75,75)
        
    if  between(cap, 8500, 9000) :
        frame = Biblur(frame,50,75,75)
        
    if  between(cap, 9000, 9500) :
        frame = Biblur(frame,9,500,75)
        
    if  between(cap, 9500, 10000) :
        frame = Biblur(frame,9,75,500)
        
    if  between(cap, 10000, 11000) :
        frame = Biblur(frame,9,1500,1500)
        
    
    return frame

def objectGrabbing_zone(frame,cap):
    if  between(cap, 12000, 19000):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bounds = np.array([26,82,119])
        upper_bounds = np.array([65,251,255])
        frame = cv2.inRange(hsv,lower_bounds,upper_bounds)
        
        
    
    if  between(cap, 15000, 19000):
        kernel = np.ones((25,25),np.uint8)
        frame = cv2.dilate(frame,kernel,  iterations = 1)
        frame = cv2.erode(frame, kernel, iterations = 1)
        sub(frame,"Close ball: Dilate + erode with kernel(25,25)")
    
    info(frame,"VHS space")
        
    
    return frame
    

def sobel_zone(frame,cap):
    
    return frame
    


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if skip(cap): #TODO remove line
                
                if cv2.waitKey(28) & 0xFF == ord('q'):
                    break
                
                if  between(cap, 1000, 3500):
                    frame = grayscale_zone(frame,cap)
                    
                if  between(cap, 5000, 8000):
                    frame = smoothing_zone(frame,cap)
                 
                if  between(cap, 8000, 11000):
                    frame = biLateral_zone(frame,cap)
                    
                if  between(cap, 12000, 50000):
                    frame = objectGrabbing_zone(frame,cap)
                    
                if  between(cap, 50000, 50000):
                    frame = sobel_zone(frame,cap)
                 
                 
    
                # (optional) display the resulting frame
                cv2.imshow('Frame', frame)
    
                # write frame that you processed to output
                out.write(frame)
                
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main("input_full.mp4", "output.mp4v")
