import cv2
import time
import os
import HandTrackingModule as htm

cap=cv2.VideoCapture(0)

wCam,hCam=640,480

cap.set(3,wCam)
cap.set(4,hCam)


folderPath="fingerImages"
myList=os.listdir(folderPath)
print(myList)

overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    resizer = cv2.resize(image, (200, 200))
    # print(image)
    overlayList.append(resizer)

# print(len(overlayList))
pTime=0

detector=htm.handDetector(detectionCon=0.70)

tipIds=[4,8,12,16,20]

while True:
    success,img=cap.read()
    img=detector.findHand(img)
    lmList=detector.findPosition(img,draw=False)
    # print(lmList)
    totalFingers=0

    if len(lmList) != 0:
        fingers=[]

        #THUMB


        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] :
            # print("Index finger is open")
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2] :
                # print("Index finger is open")
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers=fingers.count(1)
        print(totalFingers)


    h,w,c=overlayList[0].shape
    cTime=time.time()

    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f"FPS {int(fps)}",(400,70),cv2.FONT_HERSHEY_SIMPLEX,2,(200,232,0),2)
    img[0:h,0:w]=overlayList[totalFingers-1]
    cv2.rectangle(img,(28,225),(170,425),(0,255,0),cv2.FILLED)
    cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_COMPLEX_SMALL,10,(20,222,244),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)