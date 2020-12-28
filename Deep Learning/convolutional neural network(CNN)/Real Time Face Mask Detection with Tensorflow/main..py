import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np 


# img_array=cv2.imread("D:/study/datasets/maskdataset/with_mask/with_mask_3.jpg")

# # print(img_array.shape)

# # plt.imshow(img_array,cv2.COLOR_BGR2RGB)
# # plt.show()

# data_directory='D:/study/datasets/maskdataset/'

# Classes=["with_mask","without_mask"]

# for category in Classes:
#     path=os.path.join(data_directory,category)
#     for img in os.listdir(path):
#         img_array=cv2.imread(os.path.join(path,img))
#         break
#     break

# IMG_SIZE=224

# # new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
# # plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
# # plt.show()

# trainning_Data=[]

# def create_training_data():
#     for category in Classes:
#         path=os.path.join(data_directory,category)
#         class_num=Classes.index(category)
        

#         for img in os.listdir(path):
#             try:
#                 img_array=cv2.imread(os.path.join(path,img))
#                 new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#                 trainning_Data.append([new_array,class_num])
#             except  Exception as e:
#                 print(e.message)

# create_training_data()

# # print(trainning_Data[2])


# import random


# random.shuffle(trainning_Data)
# x=[]
# y=[]

# for features,label in trainning_Data:
#     x.append(features)
#     y.append(label)

# x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3)

# x=x/255.0
# y=np.array(y)

# x=x[:1000]
# y=y[:1000]


# print(x.shape,y.shape)
# print(len(x),len(y))
# print(x[999],y[999])


import pickle

# pickle_out=open('X.pickle',"wb")
# pickle.dump(x,pickle_out)
# pickle_out.close()

# pickle_out=open('Y.pickle',"wb")
# pickle.dump(y,pickle_out)
# pickle_out.close()




# pickle_in=open('X.pickle',"rb")
# x=pickle.load(pickle_in)

# pickle_in=open('Y.pickle',"rb")
# y=pickle.load(pickle_in)

# print(x.shape)
# print(y.shape)


## Transfer Learning

# model=tf.keras.applications.mobilenet.MobileNet()
# print(model.summary())

# base_input=model.layers[0].input
# base_output=model.layers[-4].output

# flat_layer=layers.Flatten()(base_output)
# final_output=layers.Dense(1)(flat_layer)
# final_output=layers.Activation('sigmoid')(final_output)

# new_model=keras.Model(inputs=base_input,outputs=final_output)
# print(new_model.summary())
# setting for binary classification (Face Mask/ with out Fasce mask)

# new_model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])


# new_model.fit(x,y,epochs=2,validation_split=0.1)

# new_model.save('my_model3.h5')


new_model=tf.keras.models.load_model('my_model3.h5')

# check network prediction

# frame=cv2.imread("D:/study/datasets/maskdataset/with_mask/with_mask_1001.jpg")

# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
# plt.show()

img_size=224
# final_image=cv2.resize(frame,(img_size,img_size))
# final_image=np.expand_dims(final_image,axis=0)
# final_image=final_image/255.0

# predict=new_model.predict(final_image)
# print(predict)

# Checking From unkown image


# frame=cv2.imread("D:/study/datasets/maskdataset/with_mask/with_mask_1001.jpg")
# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
# plt.show()
frame=cv2.imread("D:/study/datasets/maskdataset/with_mask/with_mask_1001.jpg")

faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
faces=faceCascade.detectMultiScale(gray,1.1,4)
face_roi=[]
for x,y,w,h in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    faces=faceCascade.detectMultiScale(roi_gray)
    if (len(faces) == 0):
        print('Face Not Detected')
    else:
        for (ex,ey,ew,eh)in faces:
            face_roi=roi_color[ey:ey+eh,ex:ex+ew] 

# plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
# plt.show()

# plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY))
# plt.show()


# frame=cv2.imread("D:/study/datasets/maskdataset/with_mask/with_mask_2012.jpg")
frame=cv2.imread("D:/study/datasets/maskdataset/without_mask/without_mask_2011.jpg")


final_image=cv2.resize(frame,(img_size,img_size))
final_image=np.expand_dims(final_image,axis=0)
final_image=final_image/255.0

predict=new_model.predict(final_image)
print(predict)
