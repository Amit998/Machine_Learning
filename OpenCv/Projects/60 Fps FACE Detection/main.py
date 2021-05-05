import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture("video/5.mp4")

pTime=0

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils

faceDetection=mpFaceDetection.FaceDetection(min_detection_confidence=0.8)

while True:
    success,img=cap.read()
    img=cv2.resize(img,(640,480))

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    # print(results)


    if results.detections:
        for id,detecion in enumerate(results.detections):
            # mpDraw.draw_detection(img,detecion)
            # print(id,detecion)
            # print(detecion.score)
            # print(detecion.location_data.relative_bounding_box)
            bboxC=detecion.location_data.relative_bounding_box

            ih,iw,id=img.shape

            bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                int(bboxC.width * iw),int(bboxC.height * ih),
            
            cv2.rectangle(img,bbox,(255,2,255),2)
            cv2.putText(img,f"CONFIDENCE {str(int(detecion.score[0] * 100))}%",(bbox[0],bbox[1]-50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),)


    cTime=time.time()

    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,f"FPS {str(int(fps))}",(20,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),)
    cv2.imshow("Image",img)
    cv2.waitKey(1)