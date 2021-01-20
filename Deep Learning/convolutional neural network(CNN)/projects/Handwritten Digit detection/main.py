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

print("Training Smaple dimensions",X_train_ar[0])
# print("Test Smaple dimensions",X_test_ar.shape)

from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X_train_ar.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# print('1')

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# print('2')

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# print('3')


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

# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(X_train_ar,y_train,epochs=2,validation_split=0.3)


# model.save('my_model.h5')
# import pickle

# file_name='new_model.pkl'
# pickle_save=open(file_name,'wb')
# pickle.dump(model,file_name)


model=tf.keras.models.load_model('my_model.h5')
# test_loss,test_acc=model.evaluate(X_test_ar,y_test)

# print(test_acc,'acc')

img=cv2.imread('2.png')
    

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray.shape)
resized_image=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)

# print(resized_image.shape)
new_img=tf.keras.utils.normalize(resized_image,axis=1)

new_img=np.array(new_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)

# new_img=256/new_img
print(new_img)




# predict=model.predict(new_img)

# print(np.argmax(predict))

# predictions=model.predict([X_test_ar])
# print(np.argmax(predictions[1]))
# print(y_test[1])


# print(X_test_ar[0].shape)

# print(new_img.shape)


# plt.imshow(X_test[0])
# plt.show()

# print(X_test_ar[0].shape)



# print(X_test_ar)
# print(new_img)

# print(np.argmax(predict))

# model=pickle.loads(file_name)


# predictions=model.predict(X_test_ar)

# print(np.argmax(predictions[1]))
# print(y_test[1])

# correct=0
# wrong=0

# for i in range(len(predictions)):
#     if (np.argmax(predictions[i]) == y_test[i]):
#         # print('bo yeh',i)
#         correct=correct+1
#     else:
#         wrong=wrong+1

# print(correct)
# print(wrong)
# for i in range(le)


# test_loss,test_acc=model.evaluate(predictions,y_test)

# prediction=model.predict([X_test_ar])
# print(test_loss)
# print(test_acc)

# print(np.argmax(prediction[0]))
# print(y_test[0])




# def convertImage(imgName):
#     img=cv2.imread(imgName)
    
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     resized_image=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)
#     # print(resized_image.shape)
#     new_img=tf.keras.utils.normalize(resized_image,axis=1)

#     new_img=np.array(new_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#     # print(new_img.shape)

#     return new_img

# converted_img=convertImage('2.png')
# prediction=model.predict(converted_img)
# print((prediction[0]))
# print(np.argmax(prediction))

# test_loss,test_acc=model.evaluate(X_test_ar,y_test)

# print(test_acc,'acc2')

# for i in range(10):
#     print(prediction[0][i])
# plt.imshow(converted_img[0])
# plt.show()
# print(converted_img.shape)



# def predictImg(model,conv_img):
#     print(conv_img.shape)
#     print(conv_img)
#     # print("Training Smaple dimensions",X_train_ar[0].shape)
#     # predictions=model.predict([conv_img])

#     # print(predictions)

#     # print(np.argmax(predictions))
    
# predictImg(model,converted_img)