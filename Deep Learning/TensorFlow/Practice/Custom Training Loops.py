
import os
from numpy.lib.function_base import gradient
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




num_epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
    

##Training Loop

for epoch in range(num_epochs):
    print(f"\nStart of Training Epoch {epoch} ")
    for batch_idx,(x_batch,y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred=model(x_batch,training=True),

            loss=loss_fn(y_batch,y_pred)
        
        gradients=tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(gradients,model.trainable_weights))
        acc_metric.update_state(y_batch,y_pred)

    train_acc=acc_metric.result()

    print(f"Accuracy over epoch {train_acc}")
    acc_metric.reset_states()

##TEST LOOP 
for batch_idx,(x_batch,y_batch) in enumerate(ds_test):
    y_pred=model(x_batch,training=True)
    acc_metric.update_state(y_batch,y_pred)

train_acc=acc_metric.result()
print(f"accuracy over test set")