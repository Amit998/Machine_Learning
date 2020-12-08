import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


df=pd.read_csv("Customer-Churn.csv")

df.drop('customerID',axis='columns',inplace=True)

df1=df[df.TotalCharges!=" "]
# print(df1.shape)
pd.to_numeric(df1.TotalCharges)

df1.TotalCharges=pd.to_numeric(df1.TotalCharges)

def print_unique_col_value(df):
    for col in df:
        if(df[col].dtypes=='object'):
            print(f'{col} : {df[col].unique()}') 


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
# print_unique_col_value(df1)

for col in yes_no_columns:
    df1.replace({'Yes':1,'No':0},inplace=True)


df1['gender'].replace({'Male':1,'Female':0},inplace=True)


# print(df1['gender'].unique())

df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])

scale_column=['tenure','MonthlyCharges','TotalCharges']


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df2[scale_column]=scaler.fit_transform(df2[scale_column])


# X=df2.drop('Churn',axis='columns')
# y=df2['Churn']

from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

# print(x_train.shape,y_train.shape)

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report,confusion_matrix

def ANN(x_train,y_train,x_test,y_test,loss,weights):
    model=keras.Sequential(
        [
            keras.layers.Dense(20,input_shape=(26,),activation='relu'),
            keras.layers.Dense(15,activation='relu'),
            keras.layers.Dense(1,activation='sigmoid'),
        ]
    ) 


    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    if (weights == -1):
        model.fit(x_train,y_train,epochs=100)
    else:
        model.fit(x_train,y_train,epochs=100,class_weight=weights)

    print(model.evaluate(x_test,y_test))
    y_pred=model.predict(x_test)
    y_pred=np.round(y_pred)

    print("classification Report \n ",classification_report(y_test,y_pred))

    return y_pred
# print(model.summary())


# y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)

# print(y_pred)
# yp=model.predict(x_test)

# print(df2.Churn)
count_class_0,count_class_1=df1.Churn.value_counts()


df_class_0=df2[df2['Churn'] == 0]
df_class_1=df2[df2['Churn'] == 1]


# print(df_class_0.shape)
# print(df_class_1.shape)

df_class_0_under=df_class_0.sample(count_class_1)



df_test_under=pd.concat([df_class_0_under,df_class_1],axis=0)


# print(df_test_under.Churn.value_counts())

# x=df_test_under.drop('Churn',axis='columns')
# y=df_test_under['Churn']

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)
# print(y_pred)

# print(df_test_under.Churn.value_counts())


#OVERSAMPLING

# print(df_class_1.sample(count_class_0,replace=True).shape)
df_class_1_over=df_class_1.sample(count_class_0,replace=True)
# print(df_class_1_over.shape)
df_test_over=pd.concat([df_class_0,df_class_1_over],axis=0)
# print(df_test_over.Churn.value_counts())


# x=df_test_over.drop('Churn',axis='columns')
# y=df_test_over['Churn']

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)
# y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)



#SMOTE

# x=df2.drop('Churn',axis='columns')
# y=df2['Churn']

# from imblearn.over_sampling import SMOTE

# print(y.value_counts())
# print(y.shape)

# smote=SMOTE(sampling_strategy='minority')
# x_sm,y_sm=smote.fit_sample(x,y)


# print(x_sm.shape)
# print(y_sm.shape)
# print(y_sm.value_counts())


# x_train,x_test,y_train,y_test=train_test_split(x_sm,y_sm,test_size=0.2,random_state=5,stratify=y_sm)
# y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)


# Ensemble with undersampling


# print(y_sm.value_counts())


x=df2.drop('Churn',axis='columns')
y=df2['Churn']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=15,stratify=y)

df3=x_train.copy()
df3['Churn']=y_train

df3_class_0=df3[df3.Churn==0]
df3_class_1=df3[df3.Churn==1]


# print(df3_class_0.shape,df3_class_1.shape)

df3_class_0=df3_class_0[:1495]
# print(df3_class_0.shape)

def get_train_batch(df_majority,df_minority,start,end):
    df_train=pd.concat([df_majority[start:end],df_minority],axis=0)
    x_train=df2.drop('Churn',axis='columns')
    y_train=df2['Churn']

    return x_train,y_train

# x_train,y_train=get_train_batch(df3_class_0,df3_class_1,0,1495)
# y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)

x_train,y_train=get_train_batch(df3_class_0,df3_class_1,2990,4130)
y_pred=ANN(x_train,y_train,x_test,y_test,'binary_crossentropy',-1)