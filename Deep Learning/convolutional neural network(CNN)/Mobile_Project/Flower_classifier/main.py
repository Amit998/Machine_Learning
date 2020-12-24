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



dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)


IMAGE_SIZE=224
BATCH_SIZE=64

datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator=datagen.flow_from_directory(
    data_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_generator=datagen.flow_from_directory(
    data_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)


labels='\n'.join(sorted(train_generator.class_indices.keys()))
with open('label.txt','w') as f:
  f.write(labels)

IMAGE_SHAPE=(224,224,3)

base_model=tf.keras.applications.MobileNetV2(
    IMAGE_SHAPE,
    include_top=False,
    weights='imagenet' 
)

base_model.trainable=False


model=tf.keras.Sequential(
    [
      base_model,
      tf.keras.layers.Conv2D(32,3),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(5,activation='softmax')
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epoch=10
history=model.fit(
    train_generator,
    epochs=epoch,
    validation_data=val_generator
)


saved_model_dir=''
tf.saved_model.save(model,saved_model_dir)
converter=tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model=converter.convert()


with open('model.tflite','wb') as f:
  f.write(tflite_model)