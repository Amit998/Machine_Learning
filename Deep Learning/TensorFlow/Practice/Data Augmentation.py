import os
from tensorflow.python.keras import initializers, models
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.ops.gen_array_ops import shape, split
from tensorflow.python.ops.variables import trainable_variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers

import matplotlib.pyplot
physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)
import  tensorflow_datasets as tfds


(ds_train,ds_test),ds_info=tfds.load(
    "cifar10",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# fig=tfds.show_examples(ds_train,ds_info,rows=4,cols=4)


#TF => 2.3.0 

data_augmentation=keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(height=32,width=32),
        layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
        layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ]
)


def normalize_img(image,label):
    return tf.cast(image,tf.float32)/255.0,label

def augment(image,label):
    new_height=new_weight=32
    image=tf.image.resize(image,(new_height,new_weight))

    # if(tf.random.uniform((),minval=0,maxval=1) < 0.1  ):
        # image=tf.tile(tf.image.rgb_to_grayscale(image),[1,1,3])
    image=tf.image.random_brightness(image,max_delta=0.1)
    image=tf.image.random_contrast(image,lower=0.1,upper=0.2)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_flip_up_down(image)

    return image,label


BATCH_SIZE=32

AUTOTUNE=tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train=ds_train.batch(BATCH_SIZE)
# ds_train=ds_train.map(augment,num_parallel_calls=AUTOTUNE)
# ds_train=ds_train.data
ds_train=ds_train.prefetch(AUTOTUNE)



ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(BATCH_SIZE)
ds_test=ds_test.prefetch(AUTOTUNE)


model=keras.Sequential(
    [

        keras.Input((32,32,3)),
        data_augmentation,
        layers.Conv2D(4,3,activation='relu',padding='same'),
        layers.Conv2D(8,3,activation='relu',padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(16,3,activation='relu',padding='same'),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10),
        
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


model.fit(ds_train,epochs=20,verbose=2)
print(model.evaluate(ds_test,verbose=2))