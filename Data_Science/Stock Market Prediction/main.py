import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

dataset=pd.read_csv("Google_Stock_Price_Train.csv",index_col="Date",parse_dates=True)
# dataset.head()
dataset.isna().any()
# dataset['Open'].plot(figsize=(16,6))
dataset["Close"]=dataset["Close"].str.replace(',','').astype(float)
dataset["Volume"]=dataset["Volume"].str.replace(',','').astype(float)

# print(dataset.head())
print(dataset.isna().any())
# print(dataset.info())
# dataset['Open'].plot(figsize=(16,6))
# plt.show()


# dataset['Open'].plot(figsize=(16,6))
# dataset.rolling(window=30).mean()['Close'].plot()
# plt.show()

# dataset['Close: 30 Day Mean']=dataset['Close'].rolling(window=30).mean()
# dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))
# dataset['Close'].expanding(min_periods=11).mean().plot(figsize=(16,6))
# plt.show()

training_set=dataset['Open']
training_set=pd.DataFrame(training_set)

## Feature Scalling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
trainning_set_scaled=sc.fit_transform(training_set)

X_train=[]
Y_Train=[]

for i in range(60,1250):
    X_train.append(trainning_set_scaled[i-60:i,0])
    Y_Train.append(trainning_set_scaled[i,0])
X_train,Y_Train=np.array(X_train),np.array(Y_Train)


X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor=Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))



regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))



regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,Y_Train,epochs=10,batch_size=32)

dataset_test=pd.read_csv('Google_Stock_Price_Test.csv',index_col="Date",parse_dates=True)

real_stock_Price=dataset_test.iloc[:,1:2].values
dataset_test.head()
dataset_test.info()

dataset_test["Close"]=dataset_test["Close"].str.replace(',','').astype(float)
dataset_test["Volume"]=dataset_test["Volume"].str.replace(',','').astype(float)

test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)

dataset_total=pd.concat((dataset['Open'],dataset_test['Open']),axis=0)

inputs=dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
