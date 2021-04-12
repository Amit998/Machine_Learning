import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import roc_curve
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


tf.config.experimental.list_physical_devices('GPU')
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

train_examples=20225
test_examples=2551
validation_examples=2555
img_height=img_width=224
batch_size=32

#NasNet

model=keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",trainable=True),
    layers.Dense(1,activation='sigmoid',)

])

# model.save('isic_model')

# model=keras.models.load_model("isic_model/")

# print(model.summary())

train_dataget=ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=(0.95,0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,
)

validation_datagen=ImageDataGenerator(
    rescale=1.0/255,
    dtype=tf.float32,
)
test_datagen=ImageDataGenerator(
    rescale=1.0/255,
    dtype=tf.float32,
)

train_gen=train_dataget.flow_from_directory(
    "data/train/",
    target_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123

)

validation_gen=validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123

)

test_gen=test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123

)

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    epochs=5,
    verbose=2,
    validation_data=validation_gen,
    validation_steps=validation_examples//batch_size,

)

print(model.evaluate(validation_gen,verbose=2))
print(model.evaluate(test_gen,verbose=2))