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


def get_gaze_ratio(landmarks,eye_points):
    eye_region=np.array([
            (landmarks.part(eye_points[0]).x,landmarks.part(eye_points[0]).y),
            (landmarks.part(eye_points[1]).x,landmarks.part(eye_points[1]).y),
            (landmarks.part(eye_points[2]).x,landmarks.part(eye_points[2]).y),
            (landmarks.part(eye_points[3]).x,landmarks.part(eye_points[3]).y),
            (landmarks.part(eye_points[4]).x,landmarks.part(eye_points[4]).y),
            (landmarks.part(eye_points[5]).x,landmarks.part(eye_points[5]).y),
    
        ],np.int32)
        

    height,width,_=frame.shape
    mask=np.zeros((height,width),np.uint8)

    cv2.polylines(mask,[eye_region],True,255,2)
    cv2.fillPoly(mask,[eye_region],255)
    
    eye=cv2.bitwise_and(gray,gray,mask=mask)

    min_x=np.min(eye_region[:,0])
    max_x=np.max(eye_region[:,0])
    min_y=np.min(eye_region[:,1])
    max_y=np.max(eye_region[:,1])

    gray_eye=eye[min_y:max_y,min_x:max_x]

    _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
    threshold_eye=cv2.resize(threshold_eye,None,fx=2,fy=2)

    height,width=threshold_eye.shape

    left_eye_side_threshold=threshold_eye[0:height,0:int(width/2)]
    left_eye_side_white=cv2.countNonZero(left_eye_side_threshold)


    right_eye_side_threshold=threshold_eye[0:height,int(width/2):width]
    right_side_white=cv2.countNonZero(right_eye_side_threshold)


    if left_eye_side_white == 0:
        gaze_ratio= 1
    elif right_side_white == 0:
        gaze_ratio= 5
    else:
     gaze_ratio=left_eye_side_white/right_side_white

    return gaze_ratio



while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
       
      

        landmarks=predictor(gray,face)
        gaze_ratio_left_side=get_gaze_ratio(landmarks,[36,37,38,39,40,41])
        gaze_ratio_right_side=get_gaze_ratio(landmarks,[42,43,44,45,46,47])

        gaze_ratio=(gaze_ratio_right_side+gaze_ratio_left_side)/2

        # print(gaze_ratio)

        new_frame=np.zeros((500,500,3),np.uint8)

        if gaze_ratio < 1:
            cv2.putText(frame,f"Right",(50,100),font,2,(255,255,255),3)
            print("Right")
            new_frame[:]=(0,0,255)
        elif 1 < gaze_ratio <2:
            cv2.putText(frame,f"Center",(50,100),font,2,(255,255,255),3)
            print("Center")
            new_frame[:]=(0,255,0)
        else:
            new_frame[:]=(255,0,0)
            cv2.putText(frame,f"Left",(50,100),font,2,(255,255,255),3)
            print("Left")
        

        # new_frame=np.zeros((500,500,3),np.uint8)

       




        # eye=cv2.resize(gray_eye,None,fx=2,fy=2)

        # cv2.imshow("Threshold Eye",threshold_eye)
        # cv2.imshow("Left Side Threshold Eye",left_side_threshold)
        # cv2.imshow("Right Side Threshold Eye",right_side_threshold)

        cv2.imshow("frame",frame)
        cv2.imshow("New Frame",new_frame)


    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()