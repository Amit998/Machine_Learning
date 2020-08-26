# DTR

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ls = pd.read_csv("Salary_Data.csv")
ls = pd.read_csv("new_salary.csv")
# ls.head()
x=ls.iloc[:,0:1].values

y=ls.iloc[:,1].values

from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(random_state=0)
tree_reg.fit(x,y)



# plt.scatter(x,y,color='red')
# plt.plot(x,tree_reg.predict(x),color='blue')
# plt.title("DT Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()


# x_grid=np.arange(min(x),max(x),0.1)
# x_grid=x_grid.reshape(len(x_grid),1)
# plt.scatter(x,y,color='red')
# plt.plot(x_grid,tree_reg.predict(x_grid),color='blue')
# plt.title("DT Regresssion salary vs Experience")
# plt.xlabel("Year Of Employee")
# plt.ylabel("Salary")
# plt.show()

# print(tree_reg.predict(7))
print(tree_reg.predict(np.reshape(7.6,(1,1))))