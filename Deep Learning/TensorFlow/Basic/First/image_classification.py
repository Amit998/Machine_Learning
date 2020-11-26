import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np




tf.config.experimental.list_physical_devices
print(tf.config.experimental.list_physical_devices)

print(tf.test.is_built_with_cuda())

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()

# print(x_train[0].shape)
# print(x_train)
# print(x_train)
# print(y_train.shape)

def plot_sample(index):
    plt.figure(figsize=(10,2))
    plt.imshow(x_train[index])
    plt.show()


# plot_sample(0)
# plot_sample(2)
# plot_sample(3)


classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]



# print(y_train[3])
# print(classes[y_train[3][0]])

x_train_scaled=x_train/255
x_test_scaled=x_test/255



# print(x_train_scaled[0].shape)

y_train_categorical=keras.utils.to_categorical(
    y_train,num_classes=10,dtype="float32"
)

y_test_categorical=keras.utils.to_categorical(
    y_train,num_classes=10,dtype="float32"
)

# print(y_test_categorical)

# model=keras.Sequential(
#     [keras.layers.Flatten(input_shape=(32,32,3)),
#     keras.layers.Dense(3000,activation='relu'),
#     keras.layers.Dense(1000,activation='relu'),
#     keras.layers.Dense(10,activation='sigmoid')]
# )

# model.compile(
#     optimizer='SGD',
#     loss='categorical_crossentropy',
#     metrics=['accuracy'])


# model.fit(x_train_scaled,y_train_categorical,epochs=10)


# print(model.predict(x_test_scaled)[0])



# predicted_image=np.argmax(model.predict(x_test_scaled)[0])
# print(predicted_image)
# print(classes[predicted_image])
# classes=[y_test[0][0]]



def get_model():
    model=keras.Sequential(
        [keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000,activation='relu'),
        keras.layers.Dense(1000,activation='relu'),
        keras.layers.Dense(10,activation='sigmoid')]
    )

    model.compile(
        optimizer='SGD',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


import timeit
# with tf.device('/CPU:0'):
#     start = timeit.timeit()
#     cpu_model=get_model()
#     cpu_model.fit(x_train_scaled,y_train_categorical,epochs=1)
#     end = timeit.timeit()
#     predicted_image=np.argmax(cpu_model.predict(x_test_scaled)[0])
#     print(predicted_image)
#     print(classes[predicted_image])
#     print(end - start,'s')


# with tf.device('/GUP:0'):
#     start = timeit.timeit()
#     gpu_model=get_model()
#     gpu_model.fit(x_train_scaled,y_train_categorical,epochs=1)
#     end = timeit.timeit()
#     predicted_image=np.argmax(get_model().predict(x_test_scaled)[0])
#     print(predicted_image)
#     print(classes[predicted_image])
#     print(end - start,'s')