
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height=28
img_width=28
batch_size=2




model=keras.Sequential([
    layers.Input((28,28,1)),
    layers.Conv2D(16,3,padding='same'),
    layers.Conv2D(32,3,padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])

ds_train=tf.keras.preprocessing.image_dataset_from_directory(
    'D:/study/datasets/MNIST_img/trainingSample/trainingSample/',
    labels='inferred',
    label_mode="int",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"

)
ds_validation=tf.keras.preprocessing.image_dataset_from_directory(
    'D:/study/datasets/MNIST_img/trainingSample/trainingSample/',
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"

)


def augment(x,y):
    image=tf.image.random_brightness(x,max_delta=0.05)
    return image,y

ds_train=ds_train.map(augment)

# for epochs in range(10):
#     for x,y in ds_train:
#         pass

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
#     metrics=["accuracy"]
# )
# model.fit(ds_train,epochs=10,verbose=2)

#                           METHOD 2
# ================================================================== #
#             ImageDataGenerator and flow_from_directory             #
# ================================================================== #

datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=(0.95,0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format='channels_last',
    validation_split=0.0,
    dtype=tf.float32,
)

train_generator=datagen.flow_from_directory(
    'D:/study/datasets/MNIST_img/trainingSample/trainingSample/',
    target_size=(img_height,img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=123
)

def training():pass


for epoch in range(10):
    num_batches=0

    for x,y in ds_train:
        num_batches+1
        training()

        if num_batches == 25:
            break


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"]
)
model.fit(train_generator,epochs=10,steps_per_epoch=25,verbose=2)

