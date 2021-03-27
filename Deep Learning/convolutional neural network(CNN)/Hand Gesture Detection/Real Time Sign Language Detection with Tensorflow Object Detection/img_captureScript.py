import cv2
import os
import time
import uuid

IMAGES_PATH="collectedImages"

labels=["hello",'thankyou','yes','no','iLoveYou']
num_images=1


for label in labels:
    # os.mkdir('Tensorflow\workspace\images\collectedImages\\'+label)
    cap=cv2.VideoCapture(0)
    print('Colleing Images for {}'.format(label))
    time.sleep(5)
    for imgNum in range(num_images):
        ret,frame=cap.read()
        imagename=os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))

        print(imagename)
        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(2)

        if( cv2.waitKey(1) & 0xFF == ord('q')): 
            break
            
    cap.release()

# AllImages\mask\mask.2aaa193e-8eca-11eb-9c8e-24ee9aece5f9.jpg