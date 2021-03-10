import numpy as np
import cv2
import math

capture=cv2.VideoCapture(0)

while capture.isOpened():
    ret,frame=capture.read()


    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image=frame[100:300,100:300]

    blur=cv2.GaussianBlur(crop_image,(3,3),0)

    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    mask=cv2.inRange(hsv,np.array([]))
