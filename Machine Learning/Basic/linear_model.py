import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt



df = pd.read_csv('home_price.csv')
# plt.xlabel('area')
# plt.ylabel('price')
# plt.scatter(df.area,df.price,color='red',marker='+')
# plt.show()



new_df = df.drop('price',axis='columns')
new_df


price = df.price


reg = linear_model.LinearRegression()
reg.fit(new_df,price)



# print(reg.predict([[3300]]))


m=reg.coef_
b=reg.intercept_
x=3300
y=m*x+b
# print(y)


# print(reg.coef_)

# print(reg.intercept_)


# reg.predict([[5000]])



area_df = pd.read_csv("area.csv")
# area_df.head(3)


p = reg.predict(area_df)
# print(p)



area_df['prices']=p


area_df.to_csv("prediction.csv")