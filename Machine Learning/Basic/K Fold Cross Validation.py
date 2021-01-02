from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)


#Logistic Regression

lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
# print(lr.score(X_test, y_test)) 

# SVM

svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
# print(svm.score(X_test, y_test))

#Random Forest
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
# print(rf.score(X_test, y_test))


from sklearn.model_selection import KFold

kf=KFold(n_splits=3)


# for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    # print(train_index,test_index)


def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

# print(get_score(rf,X_train,X_test,y_train,y_test))
# print(get_score(svm,X_train,X_test,y_train,y_test))
# print(get_score(lr,X_train,X_test,y_train,y_test))

from  sklearn.model_selection import StratifiedKFold
folds=StratifiedKFold(n_splits=3)

score_l=[]
score_svm=[]
score_rf=[]

for train_index,test_index in kf.split(digits.data):
    X_train,X_test,y_train,y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]

    score_rf.append(get_score(rf,X_train,X_test,y_train,y_test))
    score_svm.append(get_score(svm,X_train,X_test,y_train,y_test))
    score_l.append(get_score(lr,X_train,X_test,y_train,y_test))
 
# print(score_l,score_rf,score_svm)

from sklearn.model_selection import  cross_val_score

print(cross_val_score(lr,digits.data,digits.target))
print(cross_val_score(RandomForestClassifier(n_estimators=40),digits.data,digits.target))
print(cross_val_score(svm,digits.data,digits.target))