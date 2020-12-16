import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
import pathlib
from tensorflow.python.keras.layers.core import Dropout
import tensorflow_hub as hub

#DOWNLOAD THE DATASET


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)

IMAGE_SHAPE=(224,224)

classifier=tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])
    

print(list(data_dir.glob('*/*.jpg'))[:5])
img_count=len(list(data_dir.glob('*/*.jpg')))
# img_count

roses=list(data_dir.glob('roses/*'))
roses[:5]
flower_image_dict={
    'roses':list(data_dir.glob('roses/*')),
    'daisy':list(data_dir.glob('daisy/*')),
    'dandelion':list(data_dir.glob('dandelion/*')),
    'sunflowers':list(data_dir.glob('sunflowers/*')),
    'tulips':list(data_dir.glob('tulips/*')),
}

flower_labels_dict={
    'roses':0,
    'daisy':1,
    'dandelion':2,
    'sunflowers':3,
    'tulips':4,
}
IMAGE_SHAPE=(224,224)
test_ross=str(flower_image_dict['roses'][0])

img=cv2.imread(test_ross)
cv2.resize(img,IMAGE_SHAPE).shape

x,y=[],[]
for flower_name,images in flower_image_dict.items():
    for image in images:
        img=cv2.imread(str(image))
        resized_image=cv2.resize(img,IMAGE_SHAPE)
        x.append(resized_image)
        y.append(flower_labels_dict[flower_name])

num_classes=5

x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10)


x_train_scaled=x_train/255
x_test_scaled=x_test/255

predicted=classifier.predict(np.array([x[0],x[1],x[2]]))
# predicted

with open('labels.txt') as f:
    image_labels=f.read().splitlines()

# image_labels[722]

feature_extractor_model='https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'

pretrained_model_without_top_layer=hub.KerasLayer(
    feature_extractor_model,input_shape=(224,224,3),trainable=False
)

num_of_flowers=5

model=tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(num_of_flowers)     
])
model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(x_train_scaled,y_train,epochs=5)

model.evaluate(x_test,y_test)