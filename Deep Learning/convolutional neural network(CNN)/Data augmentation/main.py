import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
import pathlib

from tensorflow.python.keras.layers.core import Dropout


#Install the dataset

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)

print(data_dir)
image_dataset=list(data_dir.glob('*/*.jpg'))
image_count=len(image_dataset)

# print(image_dataset)
roses=list(data_dir.glob('roses/*'))


print(roses[:5])
PIL.Image.open(str(roses[2]))
flower_image_dict={
    'roses':list(data_dir.glob('roses/*')),
    'daisy':list(data_dir.glob('daisy/*')),
    'dandelion':list(data_dir.glob('dandelion/*')),
    'sunflowers':list(data_dir.glob('sunflowers/*')),
    'tulips':list(data_dir.glob('tulips/*')),
}

flower_labels_dict={
    'roses':0,
    'daisy':1,
    'dandelion':2,
    'sunflowers':3,
    'tulips':4,
}

# img=cv2.imread(str(flower_image_dict['roses'][0]))
# print(img.shape)
# print(cv2.resize(img,(180,180)).shape)

x,y=[],[]
for flower_name,images in flower_image_dict.items():
    for image in images:
        img=cv2.imread(str(image))
        resized_image=cv2.resize(img,(180,180))
        x.append(resized_image)
        y.append(flower_labels_dict[flower_name])

# print(len(x),len(y))
# print(y[0])

num_classes=5

x=np.array(x)
y=np.array(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10)


x_train_scaled=x_train/255
x_test_scaled=x_test/255
# print(len(x_train))

data_augmentation=keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.3),
    layers.experimental.preprocessing.RandomContrast(0.3),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

model=Sequential(
    [
        data_augmentation,
        layers.Conv2D(16,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(num_classes)

    ]
)

# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

# model.fit(x_train_scaled,y_train,epochs=10)

# model.save('my_model.h5')

reconstructed_model = keras.models.load_model("my_model.h5")


reconstructed_model.evaluate(x_test_scaled,y_test)

predictions=reconstructed_model.predict(x_test_scaled)

score=tf.nn.softmax(predictions[0])

# print(np.argmax(score))
# print(y_test[0])




plt.axis('off')
# plt.imshow(x[0])
# plt.show()
plt.imshow(data_augmentation(x)[0].numpy().astype("uint8"))
plt.show()