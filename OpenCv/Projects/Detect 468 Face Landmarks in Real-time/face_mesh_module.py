import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture("video/2.mp4")
pTime=0




class Face_Mesh():
    def __init__(self,static_image_mode=False,
               max_num_faces=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode
        self.max_num_faces=max_num_faces
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(max_num_faces=max_num_faces)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def get_face_loc(self,img,draw=True):
        faces=[]
        self.imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(self.imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    face=[]

                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACE_CONNECTIONS,self.drawSpec,self.drawSpec)


                    for id,lm in enumerate(faceLms.landmark):

                        ih,iw,ic=img.shape

                        x,y=int(lm.x*iw),int(lm.y*ih)
                        cv2.putText(img,f"{str(id)}",(x,y),cv2.FONT_HERSHEY_TRIPLEX,0.4,(0,200,200),)
                        face.append([id,x,y])
                    faces.append(face)
      

        return img,faces





    


def main():
    cap=cv2.VideoCapture("video/9.mp4")
    pTime=0
    detector=Face_Mesh()

    while True:
        success,img=cap.read()
        img=cv2.resize(img,(640,480))
        
        img,faces=detector.get_face_loc(img)


        
        # if len(faces) != 0:
        #     print(len(faces))
        cTime=time.time()

        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,f"FPS {str(int(fps))}",(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),)

        cv2.imshow("Image",img)
        cv2.waitKey(10)

    



if __name__ == '__main__':
    main()