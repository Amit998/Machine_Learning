import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

ls = pd.read_csv("Salary_Data.csv")
# ls.head()
x=ls.iloc[:,[0,1]].values
y=ls.iloc[:,2].values
print(x,y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=0)




from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifire=LogisticRegression(random_state=1)
classifire.fit(X_train,Y_train)

y_predict=classifire.predict(X_test)

# print(X_test)
# print(y_predict)




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_predict)
print(cm)

from sklearn.metrics import accuracy_score
ase=accuracy_score(Y_test,y_predict)
print(ase)


from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max() +1,step=0.01  ))


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