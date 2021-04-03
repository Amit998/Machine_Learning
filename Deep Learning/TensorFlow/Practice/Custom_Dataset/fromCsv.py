
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

img_height=28
img_width=28
batch_size=2


directory='data/'
df=pd.read_csv('data/train.csv')

file_path=df['file_name'].values
labels=df['label'].values
# print(file_path)

ds_train=tf.data.Dataset.from_tensor_slices((file_path,labels))

def read_image(image_file,label):
    image=tf.io.read_file(directory+image_file)
    image=tf.image.decode_image(image,channels=1,dtype=tf.float32)

    return image,label

def augment(image,label):
    return image,label


ds_train-ds_train.map(read_image).map(augment).batch(2)

for epooch in range(10):
    for x,y in ds_train:
        pass


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