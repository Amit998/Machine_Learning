# # path C:\Program Files (x86)\Tesseract-OCRC:\Program Files (x86)\Tesseract-OCR
import cv2
import numpy as np

import pytesseract
import os


per=25
pytesseract.pytesseract.tesseract_cmd="C:\\Program Files (x86)\\tesseract-OCR\\tesseract.exe"


imgQ=cv2.imread('DOCUMENT SCANNER-1.jpg')
h,w,c=imgQ.shape
imgQ=cv2.resize(imgQ,(w//3,h//3))

orb=cv2.ORB_create(1000)
kp1,des1=orb.detectAndCompute(imgQ,None)

# imgkp1=cv2.drawKeypoints(imgQ,kp1,None)
# cv2.imshow("image key point",imgkp1)


path="userForms"
myPicList=os.listdir(path)

for j,y in enumerate(myPicList):
    img=cv2.imread(path+"/"+y)
    h,w,c=img.shape
    img=cv2.resize(img,(w//3,h//3))
    # cv2.imshow(y,img)

    kp2,des2=orb.detectAndCompute(img,None)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.match(des2,des2)
    matches.sort(key=lambda x:x.distance)
    good=matches[:int(len(matches)*(per/100))]
    imgMatch=cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)

    imgMatch=cv2.resize(imgMatch,(w//3,h//3))
    cv2.imshow(y,imgMatch)


    srcPoints=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M,_=cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan=cv2.warpPerspective(img,M,(w,h))
    imgScan=cv2.resize(imgScan,(w//3,h//3))
    cv2.imshow(y,imgScan)




# print(myPicList)

# cv2.imshow("output",imgQ)
cv2.waitKey(0)