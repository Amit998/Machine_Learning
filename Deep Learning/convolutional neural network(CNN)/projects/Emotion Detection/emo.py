from PIL.Image import new
from numpy.core.fromnumeric import size
from numpy.lib import twodim_base
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np 
import pickle

# from typing_extensions import final


# test_img="D:/study/datasets/FER/train/0/Training_3908.jpg"


DATASET_PATH="D:/study/datasets/FER/train/"



# img_arr=cv2.imread(test_img)

# print(img_arr.shape)
# plt.imshow(img_arr)
# plt.show()

Classes=["0","1","2","3","4","5","6"]


# for category in Classes:
#     path=os.path.join(DATASET_PATH,category)
#     for img in os.listdir(path):
#         # print(img)
#         img_arr=cv2.imread(os.path.join(path,img))
#         # plt.imshow(cv)
#         # plt.imshow(cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB))
#         # plt.show()
#         break
#     break
# img_size=224
# new_array=cv2.resize(img_arr,(img_size,img_size))
# plt.imshow(new_array)
# plt.show()

# tranning_Data=[]
# img_size=224

# def create_training_data():
#     count=0
#     for category in Classes:
#         path=os.path.join(DATASET_PATH,category)
#         class_num=Classes.index(category)
#         for img in os.listdir(path):
#             if (count != 600):
#                 try:
#                     img_arr=cv2.imread(os.path.join(path,img))
#                     new_array=cv2.resize(img_arr,(img_size,img_size))
#                     # print(new_array,class_num)
#                     tranning_Data.append([new_array,class_num])
#                     count+=1
#                 except Exception as e:
#                     print(e)
#             else:
#                 print(class_num)
#                 print(count)
#                 count=0
#                 break
            

# create_training_data()

# # print(len(tranning_Data))

# import random

# random.shuffle(tranning_Data)

# X=[]
# Y=[]

# for features,label in tranning_Data:
#     X.append(features)
#     Y.append(label)


# X=X[:2000]
# Y=Y[:2000]

# X=np.array(X).reshape(-1,img_size,img_size,3)
# Y=np.array(Y)
# X=X/255.0

# print(X[0],Y[0])
# print(X.shape,Y.shape)





# pickle_out=open('X.pickle',"wb")
# pickle.dump(X,pickle_out)
# pickle_out.close()

# pickle_out=open('Y.pickle',"wb")
# pickle.dump(Y,pickle_out)
# pickle_out.close()

pickle_in_X=open('X.pickle',"rb")
X=pickle.load(pickle_in_X)

pickle_in_Y=open('Y.pickle',"rb")
Y=pickle.load(pickle_in_Y)



# X=X[:1000]
# Y=Y[:1000]

# print(X.shape)
# print(Y.shape)

# zero=0
# one=0
# two=0
# three=0
# four=0
# five=0
# six=0

# for i in range(len(Y)):
#     if (int(Y[i])== 0 ):
#         zero=zero+1
#     elif (int(Y[i])== 1 ):
#         one=one+1
#     elif (int(Y[i])== 2 ):
#         two=two+1
#     elif (int(Y[i])== 3 ):
#         three=three+1
#     elif (int(Y[i])== 4 ):
#         four=four+1
#     elif (int(Y[i])== 5 ):
#         five=five+1
#     elif (int(Y[i])== 6 ):
#         six=six+1

# print(zero,one,two,three,four,five,six)


# model=tf.keras.applications.MobileNetV2()

# model.summary()

# Transfer Learning -Tuning, weights start from last check post

# base_input=model.layers[0].input

# base_output=model.layers[-2].output

# final_output=layers.Dense(128)(base_output)
# final_output=layers.Activation('relu')(final_output)
# final_output=layers.Dense(64)(base_output)
# final_output=layers.Activation('relu')(final_output)
# final_output=layers.Dense(7,activation='softmax')(final_output)

# print(final_output)

# print
# new_model=keras.Model(inputs=base_input,outputs=final_output)
# print(new_model.summary())

# new_model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=["accuracy"])

# new_model.fit(X,Y,epochs=10)

new_model=tf.keras.models.load_model('my_model_3.h5')
# print(np.argmax(new_model.predict(X)))
# print(Y[0])

i=6
# X_Test=np.array(X[i]).reshape(-1,224,224,3)
# print(X_Test.shape)

# print(np.argmax(new_model.predict(X_Test)))
# print(Y[i])

count=0
for i in range(600):
    X_Test=np.array(X[i]).reshape(-1,224,224,3)
    predict=np.argmax(new_model.predict(X_Test))
    if (int(predict)==int(Y[i])):
        # print(count)
        count=count+1
print(count)

# print(Y.shape)
# for i in range(10):
    # print(X[i].shape)
    # print(np.argmax(new_model.predict(X[i])))
    # print(Y[i])

# print(X[0].shape)

# new_model.save('my_model_3.h5')

# new_model=tf.keras.models.load_model('my_model_3.h5')
# X=np.array(X).reshape(-1,img_size,img_size,3)

# X=X[:1500]
# Y=Y[:1500]



# print(X)

# print(new_model.evaluate(X,Y))
# prediction=new_model.predict(X)

# print(np.argmax(prediction))
# for i in range(10):
    # print(np.argmax(prediction[i]),Y[i])
