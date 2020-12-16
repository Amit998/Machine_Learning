from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


mnist=fetch_openml('MNIST_784')

# print(mnist)

x,y=mnist['data'],mnist['target']

# print(x.shape)

some_digit=x[36000]
some_digit_image=some_digit.reshape(28,28)

# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.show()

x_train,x_test=x[:60000],x[60000:]
y_train,y_test=y[:60000],y[60000:]

shuffle_index=np.random.permutation(60000)
x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)


y_train_2=(y_train==2)
y_test_2=(y_test==2)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(tol=0.1,solver='lbfgs')
clf.fit(x_train,y_train_2)

print(clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score

a=cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")

print(cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy"))

print(a.mean())


print(y_train[10])