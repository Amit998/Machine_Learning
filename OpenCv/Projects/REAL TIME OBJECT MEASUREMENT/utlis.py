import cv2
import numpy as  np

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgblur,cThr[0],cThr[1])
    # imgDial=cv2.dilate(imgCanny)
    kernel=np.ones((5,5))
    imgDial=cv2.dilate(img,imgCanny,kernel,iterations=3)
    imgThre=cv2.erode(imgDial,kernel,iterations=2)

    if showCanny:cv2.imshow('Canny',imgThre)

    contorous,hiearchy=cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for i in contorous:
        area=cv2.contourArea(i)
        if area >minArea:
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            bbox=cv2.boundingRect(approx)