import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import pathlib  # pathlib is in standard library

batch_size = 2
img_height = 28
img_width = 28

directory='data/'
ds_train=tf.data.Dataset.list_files(str(pathlib.Path(directory+'*.jpg')))

def process_path(file_path):
    image=tf.io.read_file(file_path)
    image=tf.io.read_file(image,channel=1)
    label=tf.strings.split(file_path,'\\')
    label=tf.strings.substr(label,pos=0,len=1)[2]
    label=tf.strings.to_number(label,out_type=tf.int64)
    return image,label

# for filepath in ds_train:
    # print(tf.strings.split(filepath,'\\')[2])
ds_train=ds_train.map(process_path).batch(batch_size)



model=keras.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(16,3,padding='same'),
    layers.Conv2D(32,3,padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"]
)
model.fit(ds_train,epochs=10,steps_per_epoch=25,verbose=2)