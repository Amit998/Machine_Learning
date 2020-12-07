import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tensorflow.keras import layers
from tensorflow.python.keras.layers.core import Dropout
warnings.filterwarnings('ignore')

df=pd.read_csv("First\sonar_dataset.csv",header=None)
# print(df.sample(10))


# print(df.isna().sum())

# print(df.columns)

x=df.drop(60,axis='columns')
y=df[60]

y=pd.get_dummies(y,drop_first=True)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
# print(x_train.shape)
# print(y_train.shape)

import tensorflow as tf
from tensorflow import keras

model=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100,batch_size=8)

# model.predict(x_test)

# print(model.evaluate(x_test,y_test))


y_pred=model.predict(x_test).reshape(-1)
print(y_pred[:5])


y_pred=np.round(y_pred)
print(y_pred[:10])
print(y_test[:10])


from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,y_pred))



model=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid'),
    keras.layers.Dropout(0.5),
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100,batch_size=8)

print(classification_report(y_test,y_pred))