import cv2
import os
import time
import uuid

IMAGES_PATH="Tensorflow/workspace/images/collectedImages"

labels=["hello",'thankyou','yes','no','iLoveYou']
num_images=15

for label in labels:
    # os.mkdir('Tensorflow\workspace\images\collectedImages\\'+label)
    cap=cv2.VideoCapture(0)
    print('Colleing Images for {}'.format(label))
    time.sleep(5)
    for imgNum in range(num_images):
        ret,frame=cap.read()
        imagename=os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(2)

        if( cv2.waitKey(1) & 0xFF == ord('q')): 
            break
            
    cap.release()
