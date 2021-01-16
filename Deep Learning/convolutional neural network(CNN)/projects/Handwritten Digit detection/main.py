import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
mnist=tf.keras.datasets.mnist


(X_train, y_train),(X_test, y_test)=mnist.load_data()

# print(X_train[0])
# plt.imshow(X_train[0],cmap=plt.cm.binary)
# print(y_train[0])
# plt.show()

#normalize the image

X_train=tf.keras.utils.normalize(X_train,axis=1)
X_test=tf.keras.utils.normalize(X_test,axis=1)
# print(X_train[0])
# plt.imshow(X_train[0],cmap=plt.cm.binary)
# plt.show()

IMG_SIZE=28
X_train_ar=np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_test_ar=np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)

# print("Training Smaple dimensions",X_train_ar.shape)
# print("Test Smaple dimensions",X_test_ar.shape)

from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X_train_ar.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

print('1')

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

print('2')

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

print('3')


#fully conneceted Layer

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(32))
model.add(Activation("relu"))


model.add(Dense(10))
model.add(Activation("softmax"))

# pint(model.summary())
# print()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train_ar,y_train,validation_split=0.3,epochs=5)


# model.save('my_model.h5')
# import pickle

# file_name='new_model.pkl'
# pickle_save=open(file_name,'wb')
# pickle.dump(model,file_name)


# model=tf.keras.models.load_model('my_model.h5')
# model=pickle.loads(file_name)

test_loss,test_acc=model.evaluate(X_test_ar,y_test)

# prediction=model.predict([X_test_ar])
# print(test_loss)
# print(test_acc)

# print(np.argmax(prediction[0]))
# print(y_test[0])




def convertImage(imgName):
    img=cv2.imread(imgName)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized_image=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
    # print(resized_image.shape)
    new_img=tf.keras.utils.normalize(resized_image,axis=1)

    new_img=np.array(new_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    # print(new_img.shape)

    return new_img

converted_img=convertImage('8.jpg')
# print(converted_img.shape)

def predictImg(model,conv_img):
    predictions=model.predict(conv_img)

    print(predictions)

    print(np.argmax(predictions))
predictImg(model,converted_img)