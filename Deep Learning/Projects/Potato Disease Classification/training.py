import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=50

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




