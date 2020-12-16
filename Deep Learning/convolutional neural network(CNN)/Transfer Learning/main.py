import numpy as np
import cv2

import PIL.Image as Image
import os

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMAGE_SHAPE=(224,224)

classifier=tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])

goldfish=Image.open('bulbul.jpg').resize(IMAGE_SHAPE)
# Image.open('goldfish.jpg')

goldfish=np.array(goldfish)/255.0
# print(goldfish.shape)

goldfish=goldfish[np.newaxis,...]
result=classifier.predict(goldfish)

result_label_index=np.argmax(result)
print(result_label_index)



with open('labled.txt') as f:
    image_labels=f.read().splitlines()

print(image_labels[result_label_index])