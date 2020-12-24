import  tensorflow as tf
import pandas as pd
import os

base_dir='D:/study/datasets/fruits-360/Training'

IMAGE_SIZE=224
BATCH_SIZE=64

datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    height_shift_range=0.3,
    width_shift_range=0.3,
    horizontal_flip=True,
    validation_split=0.1,
    fill_mode='nearest'
)

train_generator=datagen.flow_from_directory(base_dir,target_size=(IMAGE_SIZE,IMAGE_SIZE),batch_size=BATCH_SIZE,subset='training')

print(train_generator.class_indices)


labels='\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
    f.writelines(labels)

IMAGE_SHAPE=(IMAGE_SIZE,IMAGE_SIZE,3)
base_model=tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,include_top=False,weights='imagenet')


base_model.trainable=False

model=tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Conv2D(32,3,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(131,activation='softmax')
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs=10

model.fit(train_generator,epochs=epochs,steps_per_epoch=200)

converter=tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model=converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)