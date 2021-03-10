import tensorflow as tf

# print(tf.__version__)

minst=tf.keras.datasets.mnist # 28 x 28 of hand written digits 0-9



(x_train,y_train),(x_test,y_test)=minst.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))

model.compile(
            optimizer=tf.keras.optimizers.Adadelta(),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,epochs=3)

val_loss,val_acc=model.evaluate(x_test,y_test)
print(val_loss,val_acc)

model.save('path/to/location')
model = tf.keras.models.load_model('path/to/location')
# model.save('epic_num_reader_save.model')
# new_model=tf.keras.models.load_model('epic_num_reader_save.model')

predictions=model.p
print(predictions)
import numpy as np
print(np.argmax(predictions[0]))



import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.show()
plt.close()