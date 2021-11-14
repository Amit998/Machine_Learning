import numpy as np 
import pandas as pd
import os
from pandas.core.algorithms import mode
from sklearn.ensemble import RandomForestClassifier


train_data=pd.read_csv("input/train.csv")
test_data=pd.read_csv("input/test.csv")


women=train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women=sum(women)/len(women)

# print("% of women who survived",rate_women)



men=train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men=sum(men)/len(men)

# print(rate_men)

features=["Pclass","Sex","SibSp","Parch","Age"]

target=train_data['Survived']


# print(target)

X=pd.get_dummies(train_data[features].fillna(0))
X_test=pd.get_dummies(test_data[features].fillna(0))

clf=RandomForestClassifier(n_estimators=250,max_depth=7,random_state=1)
clf.fit(X,target)
predictions=clf.predict(X_test)


output=pd.DataFrame({"PassengerId":test_data.PassengerId,"Survived":predictions})


output.to_csv("my_subbmission.csv",index=False)
print(output)
