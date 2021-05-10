import cv2
import numpy as np
import dlib
import math

def midPoint(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("../face_landmarks/shape_predictor_68_face_landmarks.dat")


while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=detector(gray)

    for face in faces:
        # pass
        # x,y=face.left(),face.top()
        # x1,y1=face.right(),face.bottom()

        # cv2.rectangle(frame,(x,y),(x1,y1),(200,23,255),2)
        # print(face)
        landmarks=predictor(gray,face)
        # x=landmarks.part(36).x
        # y=landmarks.part(36).y

        # cv2.circle(frame,(x,y),3,(0,0,255),2)

        left_eye_left_point=(landmarks.part(36).x,landmarks.part(36).y)
        left_eye_right_point=(landmarks.part(39).x,landmarks.part(39).y)
        left_eye_centerTop=midPoint(landmarks.part(37),landmarks.part(38))
        left_eye_centerBottom=midPoint(landmarks.part(41),landmarks.part(40))
        left_eye_hor_line=cv2.line(frame,left_eye_left_point,left_eye_right_point,(200,200,200),2)
        left_eye_verticle_line=cv2.line(frame,left_eye_centerTop,left_eye_centerBottom,(200,200,200),2)


        right_eye_left_point=(landmarks.part(42).x,landmarks.part(42).y)
        right_eye_right_point=(landmarks.part(45).x,landmarks.part(45).y)
        right_eye_centerTop=midPoint(landmarks.part(43),landmarks.part(44))
        right_eye_centerBottom=midPoint(landmarks.part(46),landmarks.part(47))
        right_eye_hor_line=cv2.line(frame,right_eye_left_point,right_eye_right_point,(200,200,200),2)
        right_eye_verticle_line=cv2.line(frame,right_eye_centerTop,right_eye_centerBottom,(200,200,200),2)

        


    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()