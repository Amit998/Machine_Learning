import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp 

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




from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn.fit(X_train,Y_train)



y_predict=knn.predict(X_test)


print(Y_test)
print(y_predict)




from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_predict)
print(cm)

from sklearn.metrics import accuracy_score
ase=accuracy_score(Y_test,y_predict)
print(ase)


from matplotlib.colors import ListedColormap  
x_set, y_set = X_train, Y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, knn.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('red','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('red', 'green'))(i), label = j)  
mtp.title('K-NN Algorithm (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()


# from matplotlib.colors import ListedColormap
# X_set,Y_set=X_train,Y_train
# X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max() +1,step=0.01  ))


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