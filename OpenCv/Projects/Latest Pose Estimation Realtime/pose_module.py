import cv2
import mediapipe as mp
import time




class poseDetector():
    def __init__(self,mode=False,upBody=False,smooth=True,detectionCon=0.5,trackCon=0.5):

        self.mode = mode
        self.upBody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon


        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon,self.trackCon)


    def findPose(self,img,draw=True):

        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return img
    
    def getPositions(self,img,draw=True,which_part=None):
        lmList=[]
        if self.results.pose_landmarks:
            # self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape

                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
            if draw or which_part != None:
                print(len(lmList))
                print(lmList)
                # print(lmList[1][0])
                cv2.circle(img,(lmList[which_part][1],lmList[which_part][2]),10,(255,0,0),cv2.FILLED)
            
        return lmList











def main():
    cap = cv2.VideoCapture("videos/3.mp4")
    detector=poseDetector()
    pTime=0

    while True:
        success,img =cap.read()
        img = cv2.resize(img, (640, 480))
        img=detector.findPose(img,draw=True)
        lmList=detector.getPositions(img,which_part=10)
        print(lmList[10])

        cTime=time.time()

        fps=1/(cTime-pTime)
        pTime=cTime
        
        cv2.putText(img,f"{str(int(fps))}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),)
        cv2.imshow("Image",img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()