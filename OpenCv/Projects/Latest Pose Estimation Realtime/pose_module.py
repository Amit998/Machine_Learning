import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture("videos/3.mp4")

mpPose=mp.solutions.pose
pose=mpPose.Pose()

pTime=0

wCam,hCam=640,480

cap.set(3,wCam)
cap.set(4,hCam)

mDraw=mp.solutions.drawing_utils

while True:
    success,img =cap.read()
    img = cv2.resize(img, (640, 480))
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)

    # print(f"{results.pose_landmarks}")

    if results.pose_landmarks:
        mDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)

        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            # print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)



    cv2.imshow("Image",img)

    cTime=time.time()

    fps=1/(cTime-pTime)
    pTime=cTime


    cv2.putText(img,f"{str(int(fps))}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),)
    cv2.waitKey(10)

def main():
    cap = cv2.VideoCapture("videos/3.mp4")

    while True:
        success,img =cap.read()
        img = cv2.resize(img, (640, 480))


        cTime=time.time()

        fps=1/(cTime-pTime)
        pTime=cTime


if __name__ == '__main__':
    main()