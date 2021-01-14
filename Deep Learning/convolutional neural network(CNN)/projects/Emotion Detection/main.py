import  cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

img=cv2.imread('test1.jpg')

# plt.imshow(cv2.cvtColor(img,cv2.COLORS_BGR2RGB))
# plt.show()

predictions = DeepFace.analyze(img)

print(predictions)

# faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

# # cv2.CascadeClassifier()
 
# grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# faces=faceCascade.detectMultiScale(grey,1.1,4)


# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    

# font=cv2.FONT_HERSHEY_COMPLEX

# cv2.putText(img,
#             predictions['dominant_emotion'],
#             (50,50),
#             font,3,
#             (0,255,0),
#             2,
#             cv2.LINE_4
# )
# # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# # plt.show()
