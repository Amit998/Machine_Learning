import cv2
import numpy as np
import dlib
import math
import pyglet
import time
from playsound import playsound



# sound = pyglet.media.StaticSource(pyglet.media.load('pop.wav'))

sound = pyglet.media.load("pop.wav",streaming=False,)
left_sound=pyglet.media.load('left.wav',streaming=False)
right_sound=pyglet.media.load('right.wav',streaming=False)


cap=cv2.VideoCapture(0)
board=np.zeros((500,500),np.uint8)
board[:]=255

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("../face_landmarks/shape_predictor_68_face_landmarks.dat")


keyboard=np.zeros((600,1000,3),np.uint8)

keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "<"}
keys_set_2 = {0: "Y", 1: "U", 2: "I", 3: "O", 4: "P",
              5: "H", 6: "J", 7: "K", 8: "L", 9: "_",
              10: "V", 11: "B", 12: "N", 13: "M", 14: "<"}


keyboard_selected="left"

last_keyboard_selected="left"


keys_set={
    0:"Q",
    1:"W",
    2:"E",
    3:"R",
    4:"T",
    5:"A",
    6:"B",
    7:"C",
    8:"D",
    9:"E",
    10:"E",
    11:"G",
    12:"H",
    13:"I",
    14:"J",
    15:"X",
    16:"Z",
}

def midPoint(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)


font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
counter=0

text=""

def get_blinking_ratio(eye_points,faceial_landmarks):
    left_point=(faceial_landmarks.part(eye_points[0]).x,faceial_landmarks.part(eye_points[0]).y)
    right_point=(faceial_landmarks.part(eye_points[3]).x,faceial_landmarks.part(eye_points[3]).y)
    centerTop=midPoint(faceial_landmarks.part(eye_points[1]),faceial_landmarks.part(eye_points[2]))
    centerBottom=midPoint(faceial_landmarks.part(eye_points[5]),faceial_landmarks.part(eye_points[4]))

    horizontal_line_length=math.hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
    
    verticle_line_length=math.hypot((centerTop[0]-centerBottom[0]),(centerTop[1]-centerBottom[1]))

    ratio =horizontal_line_length / verticle_line_length
    return ratio

def letter(letter_index,letter,letter_light=True):

    if letter_index==0:
        x=0
        y=0
    elif (letter_index == 1):
        x=200
        y=0
    elif (letter_index == 2):
        x=400
        y=0
    elif (letter_index == 3):
        x=600
        y=0
    elif (letter_index == 4):
        x=800
        y=0
    elif (letter_index == 5):
        x=200
        y=200
    elif (letter_index == 6):
        x=400
        y=200
    elif (letter_index == 7):
        x=600
        y=200
    elif (letter_index == 8):
        x=800
        y=200
    elif (letter_index == 16):
        x=0
        y=400
    elif (letter_index == 9):
        x=200
        y=400
    elif (letter_index == 10):
        x=400
        y=400
    elif (letter_index == 11):
        x=600
        y=400
    elif (letter_index == 12):
        x=800
        y=400
    elif (letter_index == 13):
        x=200
        y=600
    elif (letter_index == 14):
        x=400
        y=800
    
    

    width=200
    height=200
    thickness=3
    font=cv2.FONT_HERSHEY_PLAIN

    text=letter
    font_scale=10
    font_th=4
    text_size=cv2.getTextSize("A",font,font_scale,font_th)[0]
    width_text,height_text=text_size[0],text_size[1]
    # print(text_size)
    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y

    if letter_light is True:
        cv2.rectangle(keyboard,(x+thickness,y+thickness),(x+width-thickness,y+height-thickness),(255,255,255),-1)
        cv2.putText(keyboard,text,(text_x,text_y),font,font_scale,(255,0,0),font_th)
    else:
        cv2.rectangle(keyboard,(x+thickness,y+thickness),(x+width-thickness,y+height-thickness),(255,0,0),-1)
        cv2.putText(keyboard,text,(text_x,text_y),font,font_scale,(255,255,255),font_th)



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


frames=0
letter_index=0

blinking_frames=0


while True:
    _,frame=cap.read()
    # keyboard[:]=(0,0,0)
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5)
    frames+=1
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:

        landmarks=predictor(gray,face)

        left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio=get_blinking_ratio([42,43,44,45,46,47],landmarks)

        blinking_ratio=(left_eye_ratio+right_eye_ratio)/2

        active_letter=keys_set[letter_index]
        # print(active_letter)

        print(blinking_ratio,'Blinking Ratio')


        if blinking_ratio > 4:
            counter+=1
            cv2.putText(frame,"Blink",(50,150),font,3,(255,0,0))
            blinking_frames+=1
            frames -=1

            print('text',text)
            print(active_letter,'active latter')

            if blinking_frames==5:
                # print('text',text)
                # print(active_letter,'active latter')
                text+=active_letter
                playsound('sound.wav')
                # sound.play()
                # time.sleep(1)
               
                # sound.stop()         
        else:
            blinking_frames=0
        # cv2.putText(frame,"Open",(50,150),font,3,(255,200,0))

        gaze_ratio_left_side=get_gaze_ratio(landmarks,[36,37,38,39,40,41])
        gaze_ratio_right_side=get_gaze_ratio(landmarks,[42,43,44,45,46,47])

        gaze_ratio=(gaze_ratio_right_side+gaze_ratio_left_side)/2

        # print(gaze_ratio)

        new_frame=np.zeros((500,500,3),np.uint8)

        if gaze_ratio < 1:
            keyboard_selected="right"
            
            if keyboard_selected != last_keyboard_selected:

                # cv2.putText(frame,f"Right",(50,100),font,2,(255,255,255),3)
               

                # right_sound.play()
                # time.sleep(1)
                playsound('right.wav')
                last_keyboard_selected=keyboard_selected
                new_frame[:]=(0,0,255)
         
        elif 1 < gaze_ratio <2:
            cv2.putText(frame,f"Center",(50,100),font,2,(255,255,255),3)
            print("Center")
            new_frame[:]=(0,255,0)
        else:
            keyboard_selected="left"
            if keyboard_selected != last_keyboard_selected:
                # left_sound.play()
                # time.sleep(1)
                last_keyboard_selected=keyboard_selected
                # new_frame[:]=(255,0,0)
                # cv2.putText(frame,f"Left",(50,100),font,2,(255,255,255),3)
                # print("Left")
                playsound('left.wav')
    
        # new_frame=np.zeros((500,500,3),np.uint8)

        # eye=cv2.resize(gray_eye,None,fx=2,fy=2)

        # cv2.imshow("Threshold Eye",threshold_eye)
        # cv2.imshow("Left Side Threshold Eye",left_side_threshold)
        # cv2.imshow("Right Side Threshold Eye",right_side_threshold)

        
    
    if frames == 15:
        letter_index+=1
        frames=0
    if letter_index == 15:
        letter_index=0
    
    for i in keys_set.keys():
        if i  ==letter_index:
            light=True
        else:
            light=False
        letter(i,keys_set[i],letter_light=light)
    
    cv2.putText(board,text,(10,100),font,4,0,3)

    cv2.imshow("frame",frame)
    # cv2.imshow("New Frame",new_frame)
    cv2.imshow("Virtual Keyboard",keyboard)
    cv2.imshow("Board",board)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()