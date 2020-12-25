import pandas as pd


data=pd.read_csv('carprices.csv')

# print(data.columns)
dummies=pd.get_dummies(data.Car_Model)

marged=pd.concat([data,dummies],axis='columns')

final=marged.drop(['Car_Model'],axis='columns')

# print(final)

from sklearn.linear_model import LinearRegression
model=LinearRegression()

x=final.drop(['Sell_Price'],axis='columns')
y=final.Sell_Price

# print(x)
# print(y)

# model.fit(x,y)


# print(model.predict([[69000,6,0,0,1]]))

# print(model.score(x,y))
from  sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

dfle=data
dfle.Car_Model=le.fit_transform(data.Car_Model)

# print(dfle)
x=dfle.drop(['Car_Model'],axis='columns')
y=dfle.Car_Model

# print(x)
# print(y)


from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

x=ohe.fit_transform(x).toarray()


model.fit(x,y)

print(model.score(x,y))