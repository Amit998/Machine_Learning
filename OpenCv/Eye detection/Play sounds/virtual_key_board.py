import cv2
import numpy as np

keyboard=np.zeros((600,1000,3),np.uint8)

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
    
    elif (letter_index == 15):
        x=0
        y=200
    elif (letter_index == 16):
        x=0
        y=400




    #keys

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

    #text settings

   


for i in keys_set.keys():
    if i % 2 ==0:
        light=True
    else:
        light=False
    letter(i,keys_set[i],letter_light=light)
 

# letter(6,"G")

cv2.imshow("keyboard",keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
