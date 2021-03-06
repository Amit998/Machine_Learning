import os

from tensorflow.python.keras.engine.training import Model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10,mnist

physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)


# (x_train,y_train),(x_test,y_test)=cifar10.load_data()

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0

# model=keras.Sequential(
#     [
#         keras.Input(shape=(32,32,3)),
#         layers.Conv2D(32,3,padding='valid',activation='relu'),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64,3,activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128,3,activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64,activation='relu'),
#         layers.Dense(10)

#     ]
# )


def my_model():
    # inputs=keras.Input(shape=(32,32,3))
    inputs=keras.Input(shape=(28,28,1))
    x=layers.Conv2D(32,3)(inputs)
    # x=layers.MaxPooling2D(pool_size=(2,3))
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,5,padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.Conv2D(128,3)(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(10)(x)
    model=keras.Model(inputs=inputs,outputs=outputs)
    return model


model=my_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)

print(model.summary())


model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2)
print(model.evaluate(x_test,y_test,batch_size=64,verbose=2))

model.save("pretrained_2")

# print(model.summary())
