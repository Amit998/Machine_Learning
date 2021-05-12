import cv2
import numpy as np


keyboard=np.zeros((900,1200,3),np.uint8)

def letter(x,y,letter):

    #keys

    width=200
    height=200
    thickness=3
    font=cv2.FONT_HERSHEY_PLAIN


    cv2.rectangle(keyboard,(x+thickness,y+thickness),(x+width-thickness,y+height-thickness),(255,0,0),thickness)

    #text settings

    text=letter
    font_scale=10
    font_th=4
    text_size=cv2.getTextSize("A",font,font_scale,font_th)[0]
    width_text,height_text=text_size[0],text_size[1]
    # print(text_size)
    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y
    cv2.putText(keyboard,text,(text_x,text_y),font,font_scale,(255,0,0),font_th)


# print(text_size)

# letter(0,0,"A")
# letter(200,0,"B")
# letter(400,0,"C")

letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letters=list(letters)
width=0
height=0
for i in range(len(letters)):
    if i % 6 == 0 and i != 0:
        width+=200
        height=200
        print(i)
    print(height,width)
    letter(height,width,letters[i])
    height+=200

cv2.imshow("Keyboard",keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()