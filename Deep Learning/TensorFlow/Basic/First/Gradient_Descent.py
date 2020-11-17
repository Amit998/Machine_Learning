import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers

df=pd.read_csv('dataset_insurance.csv')
print(df.head())


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(df[['age','affordibility']],df['bought_insurance'],test_size=0.2,random_state=2)


X_train_scaled=X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100

X_test_scaled=X_test.copy()

X_test_scaled['age']=X_test_scaled['age']/100

# print(X_test_scaled)
model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# print(len(X_train_scaled),len(y_train))
model.fit(X_train_scaled,y_train,epochs=100)
model.evaluate(X_test_scaled,y_test)


# print(model.predict(X_test_scaled))
# print(y_test)

coef,intercept=model.get_weights()
print(coef,intercept)