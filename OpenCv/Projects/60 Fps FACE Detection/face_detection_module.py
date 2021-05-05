import cv2
import mediapipe as mp
import time



class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon

        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFace(self,img,draw=True):

        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(self.imgRGB)


        if self.results.detections:
            for id,detecion in enumerate(self.results.detections):
                # self.mpDraw.draw_detection(img,detecion)
                bboxC=detecion.location_data.relative_bounding_box
                ih,iw,id=img.shape

                bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                    int(bboxC.width * iw),int(bboxC.height * ih),


                
                cv2.putText(img,f"CONFIDENCE {str(int(detecion.score[0] * 100))}%",(bbox[0],bbox[1]-50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),)     
        
        return img

    def getPoints(self,img,draw=True):
        bboxs=[]

        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(self.imgRGB)

        if self.results.detections:
            for id,detecion in enumerate(self.results.detections):
                bboxC=detecion.location_data.relative_bounding_box
                ih,iw,id=img.shape

                bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                    int(bboxC.width * iw),int(bboxC.height * ih)
                
                bboxs.append([id,bbox,detecion.score])
                img=self.fancy_draw(img,bbox)


           

        
        return img,bboxs
    
    def fancy_draw(self,img,bbox,l=30,thk=10):
        x,y,w,h=bbox
        x1,y1=x+w,y+h

        cv2.rectangle(img,bbox,(255,2,255),2)


        # Top Left
        cv2.line(img,(x,y),(x+l,y),(255,0,255),thk,)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),thk,)

         # Bottom Right

        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),thk,)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),thk,)

           # Top Right

        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),thk,)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),thk,)

        # Bottom Left

        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),thk,)
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),thk,)



        return img




def main():
    cap=cv2.VideoCapture("video/4.mp4")
    pTime=0
    detector=FaceDetector(minDetectionCon=0.2)

    while True:
        success,img=cap.read()
        img=cv2.resize(img,(640,480))

        # detector.findFace(img)
        img,bboxs=detector.getPoints(img)

        if len(bboxs) != 0:
            print(bboxs)


        cTime=time.time()

        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,f"FPS {str(int(fps))}",(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),)
        cv2.imshow("Image",img)
        cv2.waitKey(10)




if __name__ == '__main__':
    main()