import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

df = pd.read_csv('multipleReData.csv')

df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())


reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)


# print(reg.coef_)
# print(reg.intercept_)

# reg.predict([[3000, 3, 40]])

# print(round(int(reg.predict([[2500, 4, 5]])),2))


with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    mp=pickle.load(f)


print(mp.predict([[2500, 4, 5]]))

# from sklearn.externals import joblib
import joblib

joblib.dump(reg,'model_joblib')


jb=joblib.load('model_joblib')

print(jb.predict([[2500, 4, 5]]))