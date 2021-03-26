# Alright, so here have some code which should feel familiar from previous tutorials,
# here is what we want to cover
# 1. How to save and load model weights
# 2. Save and loading entire model (Serializing model)
#   - Saves weights
#   - Model architecture
#   - Training Configuration (model.compile())
#   - Optimizer and states


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


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0


model1 = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ]
)

inputs = keras.Input(784)
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)


# SavedModel format or HDF5 format
# model3 = MyModel()

# model=model1

# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(),
#     metrics=["accuracy"],
# )

# # print(x_train.shape,y_train.shape)
# # model.load_weights('checkpoint_folder/')

# model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)
# model.save_weights('checkpoint_folder/',save_format='h5')

# model.save("saved_model/") 

model=keras.models.load_model('saved_model/')
print(model.evaluate(x_test, y_test, batch_size=32, verbose=2))