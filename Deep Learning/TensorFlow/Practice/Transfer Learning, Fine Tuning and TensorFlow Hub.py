from imp import new_module
import os
from tensorflow.python.keras import initializers, models
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.variables import trainable_variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.datasets import mnist

physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)
import tensorflow_hub as hub



(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0



#USING OWN MODEL

# model=keras.models.load_model("pretrained_2")
# model.trainable=False

# # print(model.summary())


# for layer in model.layers:
#     assert  layer.trainable == False
#     layer.trainable = False

# print(model.summary())

# base_inputs=model.layers[0].input

# base_output=model.layers[-3].output

# final_output=layers.Dense(10)(base_output)

# model=keras.Model(inputs=base_inputs,outputs=final_output)

# print(model.summary())

# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

# model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
# print(model.evaluate(x_test, y_test, batch_size=32, verbose=2))



## pretrained keras models

# x=tf.random.normal(shape=(5,299,299,3))
# y=tf.constant([0,1,2,3,4])


# model=keras.applications.InceptionV3(include_top=True)

# print(model.summary())


# base_input=model.layers[0].input
# base_output=model.layers[-2].output

# final_output=layers.Dense(5)(base_output)

# new_model=keras.Model(inputs=base_input,outputs=base_output)

# new_model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

# new_model.fit(x,y,epochs=5,verbose=2)


## pretrained hub models

URL="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"

x=tf.random.normal(shape=(5,299,299,3))
y=tf.constant([0,1,2,3,4])


base_model=hub.KerasLayer(URL,input_shape=(299,299,3))
base_model.trainable=False

model=keras.Sequential(
    [
        base_model,
        layers.Dense(128,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(5),


])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)
print(model.summary())


model.fit(x,y,epochs=10,verbose=2)
