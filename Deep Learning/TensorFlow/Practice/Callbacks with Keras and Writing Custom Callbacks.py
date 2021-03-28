
import os
from tensorflow.python.keras import initializers, models
from tensorflow.python.keras.engine.training import Model
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
    "mnist",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# fig=tfds.show_examples(ds_train,ds_info,rows=4,cols=4)


# print(ds_info)


def normalize_img(image,label):
    return tf.cast(image,tf.float32)/255.0,label


BATCH_SIZE=64


AUTOTUNE=tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)

ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(128)
ds_test=ds_test.prefetch(AUTOTUNE)

model=keras.Sequential(
    [
        keras.Input((28,28,1)),
        layers.Conv2D(32,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(10),
    ]
)


save_callbacks=keras.callbacks.ModelCheckpoint(
    'checkpoint/',
    save_weights_only=True,
    monitor='accuracy',
    save_best_only=False,
)


def scheduler(epoch,lr):
    if( epoch < 2):
        return lr
    else:
        return lr * 0.99

class CustomCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs.keys())
        if (logs.get("accuracy") > 0.90):
            print("Accuracy over 90% quitting training")
            self.model.stop_training=True


lr_scheduler=keras.callbacks.LearningRateScheduler(
    scheduler,verbose=1
)


model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


model.fit(ds_train,epochs=10,verbose=2,callbacks=[save_callbacks,lr_scheduler,CustomCallBack()])
print(model.evaluate(ds_test,verbose=2))