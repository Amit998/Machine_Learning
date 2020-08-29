import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

# ls = pd.read_csv("Salary_Data.csv")

ls = pd.read_csv("new_salary.csv")
x=ls.iloc[:,0:1].values

y=ls.iloc[:,1].values
length=int(len(y))


# X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()

x=sc_x.fit_transform(x)


y=sc_y.fit_transform(np.reshape(y,(length,1)))

svr_reg=SVR(kernel='rbf')
svr_reg.fit(x,y)

# print(svr_reg.predict(np.reshape(6.6,(1,1))))
predict=sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[6.5]]))))
print(predict) 

import statsmodels.api as sm
x1=sm.add_constant(x)

reg=sm.OLS(y,x1).fit()
print(reg.summary())

# plt.scatter(x,y,color='red')
# plt.plot(x,svr_reg.predict(x),color='blue')
# plt.title("SVR salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()

