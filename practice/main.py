# import tensorflow as tf
# # print(tf.version)
# string = tf.Variable("this is a string", tf.string) 
# number = tf.Variable(324, tf.int16)
# floating = tf.Variable(1.567,tf.float64)
# rank1_tensor = tf.Variable(["Test"], tf.string) 
# rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
# tf.shape(rank2_tensor)
# tf.rank(rank1_tensor)

# import pandas as pd
# from sklearn import preprocessing
# import numpy as np

import tensorflow as tf

# data = pd.read_csv('houses_to_rent.csv',sep=',')
# print(data.head())
# print("-"*35);print("Importing Data");print("-"*35)


# data = data[['city','rooms','bathroom','parking spaces','fire insurance','furniture','rent amount']]


# lst=[]
# print(lst)

# for i in lst:
    # print(i[2:])
# newLst=[  int(x[2:].replace(',',''))   for x in lst]
# print(newLst)

# print(data.head())

# Process Data

# data['rent amount']=data['rent amount'].map(lambda  i: int(i[2:].replace(',','')))

# data['rent amount']=data['rent amount'].apply(lambda  i: i[2:] )


# print(data['rent amount'].head())
# for i in range(5):
#     lst.append(data['rent amount'][i])
    
# print(lst)
# newLst=[  int(x[2:].replace(',',''))   for x in lst]
# print(newLst)


# data['fire insurance']=data['fire insurance'].apply(lambda  x: list([str(x)]))
# print(data['fire insurance'].head())

# le=preprocessing.LabelEncoder()
# data['furniture']=le.fit_transform((data['furniture']))



# print("-"*35);print("Check Null Data");print("-"*35)

# print(data.isnull().sum())
# # data=data.dropna()

# # print(data.isnull().sum())

# print("-"*35);print("Head Data");print("-"*35)

# print(data.head())

# # Split Data

# print("-"*35);print("Split Data");print("-"*35)
# x=np.array(data.drop(['rent amount'],1))
# y=np.array(data['rent amount'])

# print('x',x.shape)
# print('y',y.shape)

# from sklearn import preprocessing,linear_model,model_selection

# xTrain,xTest,yTrain,Ytest=model_selection.train_test_split(x,y,train_size=0.9,test_size=0.1,random_state=0)
# print(xTrain.shape)
# print(yTrain.shape)

# # Tranning

# print("-"*35);print("Tranning Data");print("-"*35)

# model=linear_model.LinearRegression()
# model.fit(xTrain,yTrain)


# accuracy=model.score(xTest,Ytest)
# print('coeeficient : ',model.coef_)
# print('Intercept:',model.intercept_)

# print('accuracy', round(accuracy*100,2),'%')

# # Evaluation

# print("-"*35);print("Manual Testing Data");print("-"*35)

# testValues=model.predict(xTest)
# # print(testValues.shape)
# error=[]
# for i,testVal in enumerate(testValues):

#     error.append(Ytest[i]-testVal)
#     print(f'Actual values:',{Ytest[i]},'Prediction: ',{int(testVal)},' Error: ',round(error[i],2))