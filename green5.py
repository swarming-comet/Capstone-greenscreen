import numpy as np
import cv2
import sys
from numpy import zeros, newaxis

#Webcam
cap = cv2.VideoCapture(0)
#Creating mask for Background Subtraction
fgbg = cv2.createBackgroundSubtractorMOG2(150, 12, False)
(grabbed, frame) = cap.read()
snapshot = np.zeros(frame.shape, dtype=np.uint8)
#Backgrounds
background = cv2.imread('C:/Users/chris/Desktop/capstone/grand.png')
background2 = cv2.imread('C:/Users/chris/Desktop/capstone/pg.png',0)
#Resizing of background
crop_background = background[0:640, 0:480]
crop_background = cv2.resize(crop_background,(640,480))

while(1):
    ret, frame = cap.read()
    #Applying the mask to the webcam capture
    fgmask = fgbg.apply(frame,1.0)
    fgmask2 = fgbg.apply(snapshot)
    #Displays video feed
    cv2.imshow('frame',frame)
    #Mask
    cv2.imshow('fgmask',fgmask)
    replaced_image = cv2.bitwise_and(frame,frame,mask = fgmask)
    cv2.imshow('video', replaced_image)
    replaced_image3 = cv2.bitwise_and(frame,frame,mask = fgmask2)
    cv2.imshow('video3', replaced_image3)
    #Putting in the green screen element 
    replaced_image[np.where((replaced_image==[0,0,0]).all(axis=2))] = [0,255,0]
    #Display the green screen element 
    cv2.imshow('video4', replaced_image)
    #Singling out the green
    lower_green = np.array([0,254,0])
    upper_green = np.array([0,255,0])
    mask = cv2.inRange(replaced_image, lower_green, upper_green)
    #Replacing the green
    replaced_image3 = cv2.bitwise_and(crop_background,crop_background,mask = mask)
    replaced_image = replaced_image3 + replaced_image
    #Final Display
    cv2.imshow('video5', replaced_image)

    #Escape
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('t'):
        snapshot = frame.copy()
        cv2.imshow('Snapshot', snapshot)
        replaced_image2 = cv2.bitwise_and(snapshot,snapshot,mask = fgmask)
        cv2.imshow('video2', replaced_image2)
        

    

cap.release()
cv2.destroyAllWindows()