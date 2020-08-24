import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ls = pd.read_csv("Salary_Data.csv")
# ls.head()
x=ls.iloc[:,0:1].values

y=ls.iloc[:,1].values

# print(x)



# X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)


lin_reg=LinearRegression()
lin_reg.fit(x,y)



from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

# # For Prediction

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


# lin_reg2.fit(x_poly,y)
# y_predict_2=lin_reg2.predict(x)
# print(y_predict_2)



# Linear

# plt.scatter(x,y,color='red')
# plt.plot(x,lin_reg.predict(x),color='blue')
# plt.title("Polynominal Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()

# Polynominal

# plt.scatter(x,y,color='red')
# plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
# plt.title("Polynominal Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()

# y_predict=lin_reg.predict(6.5)
# print(y_predict)

lin_reg2.predict(poly_reg.fit_transform(6.7))