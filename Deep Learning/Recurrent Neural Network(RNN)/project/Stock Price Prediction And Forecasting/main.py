import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error



key="f36f43ea6154ad9f8d14295c8da141e1063ac074"

# df=pdr.get_data_tiingo('AAPL',api_key=key)

# df.to_csv('AAPL1.csv')


df=pd.read_csv('AAPL.csv',sep=',')

# print(df.tail())
# print(df.columns)
# print(df.shape)


df1=df.reset_index()['close']

# print(df1.head())

# plt.plot(df1)
# plt.show()

scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

# print(df1)
# print(df1.shape)


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# print(len(train_data),len(test_data))

def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    
    return np.array(dataX),np.array(dataY)

time_step=100

X_train,Y_train=create_dataset(train_data,time_step)
X_test,Y_test=create_dataset(test_data,time_step)


# print(X_train,'\n',Y_train)

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# print(X_train.shape)
# print(X_test.shape)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

# model.summary()
# model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=64,verbose=1)

# model.save('model_saved.h5')

model=tf.keras.models.load_model('model_saved.h5')

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

# print(len(train_predict))
# print(len(test_predict))

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

# print(math.sqrt(mean_squared_error(Y_train,train_predict)))


# print(math.sqrt(mean_squared_error(Y_test,test_predict)))


# look_back=100
# trainPredictPlot=np.empty_like(df1)
# trainPredictPlot[:,:]=np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict

# testPredictPlot=np.empty_like(df1)
# testPredictPlot[:,:]=np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict


# plt.plot(scaler.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


# print(len(test_data))

x_input=test_data[341:].reshape(1,-1)
# print(x_input.shape,'here')

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
# print(len(temp_input))

lst_output=[]
n_steps=100
i=0

while (i<30):
    if (len(temp_input)>100):
        print(len(temp_input),'temp')
        # print(i)
        x_input=np.array(temp_input[1:])
        print(len(x_input),'x')
        # print("{} day input {} ".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))

        ythat=model.predict(x_input,verbose=0)
        # print("{} day output {}".format(i,ythat))
        temp_input.extend(ythat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        ythat=model.predict(x_input,verbose=0)
        # print(ythat[0])
        temp_input.extend(ythat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(ythat.tolist())
        i=i+1
        

# print(lst_output)

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

df3=df1.tolist()
df3.extends(lst_output)

# plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
# plt.plot(day_pred,scaler.incerse_transform(lst_output))
# plt.show()

plt.plot(df3[1000:])
plt.show()