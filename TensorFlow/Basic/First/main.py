import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.python.keras.backend import flatten


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


x_train=x_train / 255
x_test=x_test / 255
# print(y_train[:4])
# plt.matshow(y_train[20])
# plt.show()

# print(x_train.shape)

x_train_flattened=x_train.reshape(len(x_train),28 * 28)
x_test_flattened=x_test.reshape(len(x_test),28 * 28)
# print(x_test_flattened.shape)


# model=keras.Sequential([
#     keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
    
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     x_train_flattened,
#     y_train,
#     epochs=5
# )


## Added A Hidden Layer

# model=keras.Sequential([
#     keras.layers.Dense(10,input_shape=(784,),activation='relu'),
#     keras.layers.Dense(10,activation='sigmoid') #Hidden Layer
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(
#     x_train_flattened,
#     y_train,
#     epochs=5
# )


## Added Flattern array using keras

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid') #Hidden Layer
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train_flattened,
    y_train,
    epochs=5
)

model.evaluate(x_test_flattened,y_test)
y_pred=model.predict(x_test_flattened)
print(y_pred[0])
# plt.matshow(y_pred[0])
# plt.matshow(x_test[3])
# print(np.argmax(y_pred[3]))
# plt.show()

y_predicted_labels=[np.argmax(i) for i in y_pred]

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()