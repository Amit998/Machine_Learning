import cv2


cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1288)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)



while True: 
    _,frame=cap.read()
    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)


    height,width,_=frame.shape


    cx=int(width/2)
    cy=int(height/2)

    pixel_center=hsv_frame[cy,cx]
    h,s,v=hsv_frame[cy,cx]
    color="undefined"

    if h < 5:
        color="red"
    elif h < 22:
        color="Orange"
    elif h < 33:
        color="Yellow"
    elif h < 78:
        color="Green"
    elif h < 131:
        color="blue"
    elif h < 167:
        color="violet"
    else:
        color="undefined"
    




    b,g,r=frame[cy,cx]
    cv2.putText(frame,color,(10,50),0,1,(int(b),int(g),int(r)),2)
    cv2.circle(frame,(cx,cy),5,(255,0,0),3)


    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)

    if key==27:
        break

cap.release()
cv2.destroyAllWindows()