# import pandas as pd

# df=pd.read_csv('carprices.csv')

# inputs=df.drop('Sell_Price',axis='columns')

# targets=df['Sell_Price']

# from sklearn.preprocessing import LabelEncoder

# le_car_model=LabelEncoder()

# inputs['car_model_n']=le_car_model.fit_transform(inputs['Car_Model'])

# inputs_n=inputs.drop(['Car_Model'],axis="columns")

# # print(inputs_n)

# from sklearn import tree

# model=tree.DecisionTreeClassifier()

# model.fit(inputs_n,targets)

# # print(model.score(inputs_n,targets))

# print(model.predict([[69000,6,1]]))

import pandas as pd
from scipy.sparse.construct import random
import sklearn

df=pd.read_csv('titanic.csv',sep=',')

# print(df.shape)

from sklearn.preprocessing import LabelEncoder

Name_l_encoder=LabelEncoder()
Sex_l_encoder=LabelEncoder()
Embarked_l_encoder=LabelEncoder()

df['Name_n']=Name_l_encoder.fit_transform(df['Name'])
df['Sex_n']=Sex_l_encoder.fit_transform(df['Sex'])
df.dropna(subset=["Age"],inplace=True)
inputs=df.drop(['Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','Survived'],axis='columns')
# print(target.head())
target=df['Survived']


inputs['Fare']=inputs['Fare'].apply(lambda x: int(x) )





inputs['Age']=inputs['Age'].apply(lambda x: int(x) )
# print(inputs['Age'].unique())
# print(inputs.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=42)

# print(x_train.shape,y_train.shape)
from sklearn import tree

model=tree.DecisionTreeClassifier()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))
# input()