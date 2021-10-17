import cv2
import numpy as np
from object_detection import ObjectDetection
import math


#initialize object detection
ob=ObjectDetection()



cap=cv2.VideoCapture("los_angeles.mp4")


#initialize counter
count=0

center_point=[]
center_pts_current_frame=[[0,0]]
center_pts_prev_frame=[[0,0]]
tracking_objects={}


tracking_id=0

while True:
    ret,frame=cap.read()
    count+=1

    if not ret:
        break

    # Detect object on frame


    (class_id,score,bboxes)=ob.detect(frame)

    for box in bboxes:
        (x,y,w,h)=box

        cx=int((x+x+w)/2)
        cy=int((y+y+h)/2)
        center_pts_current_frame.append((cx,cy))

        # cv2.circle(frame,(cx,cy),5,(0,0,255),-1)


        # print(f"FRAME NUMBER {count} BOX",x,y,w,h)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(22,22,22),3)
    
    # for pt in center_pts_current_frame:
    #     cv2.circle(frame,pt,5,(0,0,255),-1)

    # cv2.imshow("Frame",frame)

    print("CUR FRAME")
    print(center_pts_current_frame[0])

    print("Prev FRAME")
    print(center_pts_prev_frame[0])


    for pt in center_pts_current_frame:
        for pt2 in center_pts_prev_frame:
            distances=math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])

            if distances < 10:
                tracking_objects[tracking_id]=pt
                tracking_id+=1
    

    for object_id in tracking_objects.items():
        continue
    

    print("Tracking Objects")
    print(tracking_objects)







    #make a copy of the points
    center_pts_prev_frame=center_pts_current_frame.copy()




    key=cv2.waitKey(10)

    if key== 27:
        break

cap.release()
cv2.destroyAllWindows()