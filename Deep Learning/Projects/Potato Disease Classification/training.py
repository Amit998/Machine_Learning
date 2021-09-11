import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=1

print(tf.config.list_physical_devices('GPU'))

dataset=tf.keras.preprocessing.image_dataset_from_directory(
    "D:\study\datasets\PlantVillage\potato",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


class_names=dataset.class_names
# print(class_names)

# for image_batch,label_batch in dataset.take(1):

#     plt.imshow(image_batch[0].numpy().astype("uint8"))
#     plt.axis("off")
#     plt.show()
#     print(image_batch.shape)
#     print(label_batch.numpy())


# train_size=0.8
# train_num=int(len(dataset)*train_size)



# train_ds=dataset.take(train_num)
# print(len(train_ds))


# val_size=0.1
# val_num=int(len(dataset)*val_size)


# val_ds=dataset.take(val_num)


def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=1000):
    ds_size=len(ds)

    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)


    train_size=int(train_split * ds_size)
    val_size=int(val_split * ds_size)


    train_ds=ds.take(train_size)
    test_ds=ds.skip(train_size).skip(val_size)
    val_ds=ds.skip(train_size).take(val_size)

    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)

# print(len(train_ds),len(val_ds),len(test_ds))


train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])



data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])



input_shape_=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=3
model=models.Sequential(
    [
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape_),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(n_classes,activation='softmax')
    ]
)

model.build(input_shape=input_shape_)

print(model.summary())


print(input_shape_)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
)

model_version=1


import pickle
 

Pkl_Filename = "model.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(model, file)


history = "history.pkl"  

with open(history, 'wb') as file:  
    pickle.dump(history, file)



model.save(f"model-{model_version}")

model.save('/models/')
scores=model.evaluate(test_ds)

print(scores)


print(history.history.keys())



# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']



# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(range(EPOCHS), acc, label='Training Accuracy')
# plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(range(EPOCHS), loss, label='Training Loss')
# plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# for image_batch,labels_batch in test_ds.take(1):
#     first_image=(image_batch[0].numpy().astype('uint8'))
#     first_label=labels_batch[0]

#     print("First Image to predict")
#     print("First Label was  ",class_names[first_label])
#     batch_preddiction=model.predict(image_batch)
#     predicted_index=np.argmax(batch_preddiction[0])
#     print("First predicted Label was  ",class_names[predicted_index])



#     plt.show(first_image)



# def predict(model,image):
#     image_array=tf.keras.preprocessing.image.img_to_array(image[i].numpy())
#     image_array=tf.expand_dims(image_array,0)


#     predictions=model.predict(image_array)

#     predicted_class=class_names[np.argmax[predictions[0]]]
#     confidence=round(100*(np.max(predictions[0])),2)

#     return predicted_class,confidence

# plt.figure(figsize=(15,15))
# for images,labels in test_ds.take(1):
#     for i in range(9):
#         ax=plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         predicted_class,confidence=predict(model,images[i].numpy())

#         actual_class=class_names[labels[i]]

#         plt.title(f"Actual: {actual_class} \n Predictedt class : {predicted_class} \n Confidence {confidence}")

#         plt.axis("off")




