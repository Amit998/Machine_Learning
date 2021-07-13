import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os



import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array,load_img

inception_weight_path="https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"

epoch=1e-3
batch_size=128
data=[]
labels=[]

img_width=224
img_height=224
size=224


#Image Data Generator


train_datagen=ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range=20,
    rescale=1/255,
)


test_datagen=ImageDataGenerator(
    rescale=1/255,
)

target_size=(size,size)


train_generator=train_datagen.flow_from_directory(
    directory="D:/study/datasets/male-female-face-dataset-main/male-female-face-dataset-main/Training",
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator=train_datagen.flow_from_directory(
    directory="D:/study/datasets/male-female-face-dataset-main/male-female-face-dataset-main/Validation",
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)




# print(train_generator.class_mode)

x,y=train_generator.next()


# print(x)

#build model

model=Sequential()
# model.add(InceptionV3(include_top=False,pooling="avg",weights=inception_weight_path))
model.add(InceptionV3(include_top=False,pooling="avg",weights='imagenet'))
model.add(Flatten())


model.add(BatchNormalization())

model.add(Dense(2048,activation='relu'))
model.add(BatchNormalization())



model.add(Dense(1024,activation='relu'))
model.add(BatchNormalization())


model.add(Dense(1,activation='sigmoid'))

model.layers[0].trainable=False



checkpoint_path = "training_1/cp.ckpt"

os.makedirs(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




model.fit(train_generator,steps_per_epoch=len(train_generator.filenames)//batch_size,epochs=5,validation_data=validation_generator,validation_steps=len(validation_generator.filenames)//batch_size,callbacks=[cp_callback])

#test model


test_image_path="D:/study/datasets/male-female-face-dataset-main/male-female-face-dataset-main/Validation/female/112944.jpg.jpg"


#load model
latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(checkpoint_path)


def preprocess_image(test_image_path):
    img=load_img(test_image_path,target_size=(size,size,3))
    img=img_to_array(img)
    img=img/255.0
    img=img.reshape(1,size,size,3)
    return img



img=preprocess_image(test_image_path)


def get_class(data):
    pred=model.predict(data)[0][0]
    # pred=np.argmax(prob)

    if pred<=0.5:
        return 'female', (1-pred)
    else:
     return 'male', pred




prediction,acc=get_class(img)



#RealTime Using Webcam