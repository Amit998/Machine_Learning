import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers

# age=[22,25,47,52,46,56,27]
# affordabiity=[1,0,1,0,1,1,0]
# have_insurance=[0,0,1,0,1,1,0]



df=pd.read_csv('dataset_insurance.csv')
print(df.head())


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(df[['age','affordibility']],df['bought_insurance'],test_size=0.2,random_state=2)


X_train_scaled=X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100

X_test_scaled=X_test.copy()

X_test_scaled['age']=X_test_scaled['age']/100

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


# print(model.predict(X_test_scaled))
# print(y_test)

# coef,intercept=model.get_weights()




# def sigmoid(x):
#     import math
#     return 1 / (1 + math.exp(-x))



# def prediction_function(age,affordibility):
#     weighted_sum= coef[0] * age + coef[1] * affordibility + intercept
#     return sigmoid(weighted_sum)


# for i,j in zip(age,affordabiity):
    
#     # print(i,j)
#     print(prediction_function(i/100,j))

def log_loss(y_test,y_pred):
    epsilon=1e-15
    y_pred_new=[max(i,epsilon) for i in y_pred]
    y_pred_new=[min(i,1-epsilon) for i in y_pred_new]
    y_pred_new=np.array(y_pred_new)
    return -np.mean(y_test*np.log(y_pred_new)+(1-y_test) * np.log(1-y_pred_new))


def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))

# print(sigmoid_numpy(np.array([12,0,1])))

def gradient_descent(age,affordibility,y_train,epochs,loss_thrashold):
    w1=w2=1
    bias=0
    rate=0.5
    n=len(age)

    for i in range(epochs):
        wighted_sum=w1 * age + w2 * affordibility + bias
        y_pred=sigmoid_numpy(wighted_sum)
        loss=log_loss(y_train,y_pred)

        wd1=(1/n)*np.dot(np.transpose(age),(y_pred-y_train))
        wd2=(1/n)*np.dot(np.transpose(affordibility),(y_pred-y_train))

        bias_d=np.mean(y_pred-y_train)

        w1=w1 - rate * wd1
        w2 = w2 - rate * wd2
        bias=bias - rate * bias_d

        print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
        if( loss <= loss_thrashold):
            return w1,w2,bias
    return w1,w2,bias

# coefW1,coefW2 ,intercept= gradient_descent(X_train_scaled['age'],X_train_scaled['affordibility'],y_train,1000,0.4631)

coef= gradient_descent(X_train_scaled['age'],X_train_scaled['affordibility'],y_train,1000,0.4631)


# print(coefW1,coefW2,intercept)
# print(coef[0])


def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))




def prediction_function(age,affordibility):
    weighted_sum= coef[0] * age + coef[1] * affordibility + coef[2]
    return sigmoid(weighted_sum)

print(prediction_function(.30,0))