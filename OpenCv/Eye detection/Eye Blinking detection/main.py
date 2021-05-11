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
# left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)

def get_blinking_ratio(eye_points,faceial_landmarks):
    left_point=(faceial_landmarks.part(eye_points[0]).x,faceial_landmarks.part(eye_points[0]).y)
    right_point=(faceial_landmarks.part(eye_points[3]).x,faceial_landmarks.part(eye_points[3]).y)
    centerTop=midPoint(faceial_landmarks.part(eye_points[1]),faceial_landmarks.part(eye_points[2]))
    centerBottom=midPoint(faceial_landmarks.part(eye_points[5]),faceial_landmarks.part(eye_points[4]))

    horizontal_line_length=math.hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    
    verticle_line_length=math.hypot((centerTop[0]-centerBottom[0]),(centerTop[1]-centerBottom[1]))

    # print(horizontal_line_length,verticle_line_length)
    # print(centerTop,centerBottom)
    ratio =horizontal_line_length / verticle_line_length
    return ratio


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

        # left_eye_left_point=(landmarks.part(36).x,landmarks.part(36).y)
        # left_eye_right_point=(landmarks.part(39).x,landmarks.part(39).y)
        # left_eye_centerTop=midPoint(landmarks.part(37),landmarks.part(38))
        # left_eye_centerBottom=midPoint(landmarks.part(41),landmarks.part(40))
        # left_eye_hor_line=cv2.line(frame,left_eye_left_point,left_eye_right_point,(200,200,200),2)
        # left_eye_verticle_line=cv2.line(frame,left_eye_centerTop,left_eye_centerBottom,(200,200,200),2)


        # right_eye_left_point=(landmarks.part(42).x,landmarks.part(42).y)
        # right_eye_right_point=(landmarks.part(45).x,landmarks.part(45).y)
        # right_eye_centerTop=midPoint(landmarks.part(43),landmarks.part(44))
        # right_eye_centerBottom=midPoint(landmarks.part(46),landmarks.part(47))
        # right_eye_hor_line=cv2.line(frame,right_eye_left_point,right_eye_right_point,(200,200,200),2)
        # right_eye_verticle_line=cv2.line(frame,right_eye_centerTop,right_eye_centerBottom,(200,200,200),2)

        # print(left_eye_centerBottom[0])

        # horizontal_line_length=math.hypot((left_eye_left_point[0]-left_eye_right_point[0]),(left_eye_left_point[1]-left_eye_right_point[1]))

        
        # ver_line_length=math.hypot((left_eye_centerTop[0]-left_eye_centerBottom[0]),(left_eye_centerTop[1]-left_eye_centerBottom[1]))

        # ratio=horizontal_line_length/ver_line_length

        # print(horizontal_line_length,ver_line_length)
        # print(ratio)

        # if ratio > 3.5:
        #     cv2.putText(frame,"Closed",(50,150),font,3,(255,0,0))
        # else:
        #     cv2.putText(frame,"Open",(50,150),font,3,(255,200,0))

        # landmarks=predictor(gray,face)

        left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio=get_blinking_ratio([42,43,44,45,46,47],landmarks)

        blinking_ratio=(left_eye_ratio+right_eye_ratio)/2

        print(left_eye_ratio,"left Eye Ratio")
        print(right_eye_ratio, "Right Eye Ratio")
        print(blinking_ratio,'Both Ratio')

        # print(left_eye_ratio)

        # print(landmarks.part(43).x)
        
        if blinking_ratio > 5.5:
            counter+=1
            # cv2.putText(frame,"Blink",(50,150),font,3,(255,0,0))
        # else:
        #     cv2.putText(frame,"Open",(50,150),font,3,(255,200,0))


        print(counter,"Blink")

        


    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()