import cv2
import numpy as np
import dlib
import math

def midPoint(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("../face_landmarks/shape_predictor_68_face_landmarks.dat")
font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
counter=0




while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
       
        # Gaze DETECTION
        landmarks=predictor(gray,face)

        left_eye_region=np.array([
            (landmarks.part(36).x,landmarks.part(36).y),
            (landmarks.part(37).x,landmarks.part(37).y),
            (landmarks.part(38).x,landmarks.part(38).y),
            (landmarks.part(39).x,landmarks.part(39).y),
            (landmarks.part(40).x,landmarks.part(40).y),
            (landmarks.part(41).x,landmarks.part(41).y),
    
        ],np.int32)

        height,width,_=frame.shape
        mask=np.zeros((height,width),np.uint8)

        cv2.polylines(mask,[left_eye_region],True,255,2)
        cv2.fillPoly(mask,[left_eye_region],255)
        
        left_eye=cv2.bitwise_and(gray,gray,mask=mask)

        min_x=np.min(left_eye_region[:,0])
        max_x=np.max(left_eye_region[:,0])
        min_y=np.min(left_eye_region[:,1])
        max_y=np.max(left_eye_region[:,1])

        gray_eye=left_eye[min_y:max_y,min_x:max_x]

        _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
        threshold_eye=cv2.resize(threshold_eye,None,fx=2,fy=2)
        eye=cv2.resize(gray_eye,None,fx=2,fy=2)
        cv2.imshow("Eye",eye)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()