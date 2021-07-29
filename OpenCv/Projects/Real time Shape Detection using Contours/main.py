import cv2
import numpy as np



frameWidth=640
frameHeight=400


cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)


def empty(a):
    pass
cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",640,240)
cv2.createTrackbar("threshold1","parameters",23,255,empty)
cv2.createTrackbar("threshold2","parameters",20,255,empty)
cv2.createTrackbar("Area","parameters",5000,30000,empty)




def getContours(img,imageContours):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imageContours,contours,-1,(255,0,255),7)
    areaMin=cv2.getTrackbarPos("Area","parameters")
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area > areaMin:
            cv2.drawContours(imageContours,cnt,-1,(255,0,255),7)

            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt, 0.02 * peri, True)

            length=len(approx)
            x,y,w,h=cv2.boundingRect(approx)

            cv2.rectangle(imageContours,(x,y),(x+w,y+h),(0,255,0),5)

            cv2.putText(imageContours,"Points :"+str(len(approx)),(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),3)

            cv2.putText(imageContours,"area :"+str(int(area)),(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),3)

            print(length)


    return




while True:
    success,img=cap.read()
    imageContours=img.copy()

    imgBlur=cv2.GaussianBlur(img,(7,7),1)
    imgGray=cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
    threshold1=cv2.getTrackbarPos("threshold1","parameters")
    threshold2=cv2.getTrackbarPos("threshold12","parameters")
    kernel=np.ones((5,5))
    
    imgCanny=cv2.Canny(imgGray,threshold1,threshold2)
    imgDil=cv2.dilate(imgCanny,kernel,iterations=1)

    getContours(imgDil,imageContours)






    cv2.imshow("Canny",imgCanny)
    cv2.imshow("image contours",imageContours)
    # cv2.imshow("result",imgGray)
    # cv2.imshow("result",imgBlur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break