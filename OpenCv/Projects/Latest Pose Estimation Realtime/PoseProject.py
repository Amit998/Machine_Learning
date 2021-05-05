import cv2
from pose_module import poseDetector
import time

cap = cv2.VideoCapture("videos/5.mp4")
detector=poseDetector()
pTime=0

while True:
    success,img =cap.read()
    img = cv2.resize(img, (640, 480))
    img=detector.findPose(img,draw=True)
    lmList=detector.getPositions(img,which_part=10)
    # print(lmList[10])

    cTime=time.time()

    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,f"{str(int(fps))}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),)
    cv2.imshow("Image",img)
    cv2.waitKey(10)