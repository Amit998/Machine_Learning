# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from tensorflow.keras import layers

# df=pd.read_csv('dataset_insurance.csv')
# print(df.head())


# from sklearn.model_selection import train_test_split


# X_train,X_test,y_train,y_test=train_test_split(df[['age','affordibility']],df['bought_insurance'],test_size=0.2,random_state=2)


# X_train_scaled=X_train.copy()
# X_train_scaled['age']=X_train_scaled['age']/100

# X_test_scaled=X_test.copy()

# X_test_scaled['age']=X_test_scaled['age']/100

# # print(X_test_scaled)
# model=keras.Sequential([
#     keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
# ])

# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # print(len(X_train_scaled),len(y_train))
# model.fit(X_train_scaled,y_train,epochs=100)
# model.evaluate(X_test_scaled,y_test)


# # print(model.predict(X_test_scaled))
# # print(y_test)

# coef,intercept=model.get_weights()
# print(coef,intercept)


# Step_Size=slope * learning_rate

import numpy as np


def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=10000
    learning_rate=0.008
    n=len(x)
    for i in range(iterations):
        y_pred=m_curr*x+b_curr
        const=(1/n) * sum([val**2 for val in (y-y_pred)])
        md=-(2/n) * sum(x*(y-y_pred))
        bd=-(2/n) * sum(y-y_pred)
        m_curr=m_curr - learning_rate * md
        b_curr=b_curr -learning_rate * bd

        print("M  {} , B: {}, iteration: {}, const{}".format(m_curr,b_curr,i,const))

x=np.array([1,2,3,4,5,6])
y=np.array([5,7,9,11,13,15])
gradient_descent(x,y)