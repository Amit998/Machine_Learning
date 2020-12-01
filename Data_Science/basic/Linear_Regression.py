import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

ls = pd.read_csv("Salary_Data.csv")
# ls.head()
x=ls.iloc[:,:-1].values
y=ls.iloc[:,1].values
# print(x,y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_predict=reg.predict(X_test)

# print(y_predict)
# print(Y_test)
# print(y_predict)
# for i in Y_test:
#     print(i)

# for j in y_predict:
#     print(j)
# plt.scatter(X_train,Y_train,color='red')
# plt.plot(X_train,reg.predict(X_train),color='blue')
# plt.title("Linear Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()

# plt.scatter(X_test,Y_test,color='red')
# plt.plot(X_train,reg.predict(X_train),color='blue')
# plt.title("Linear Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()