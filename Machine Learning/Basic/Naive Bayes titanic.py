import pandas as pd



df=pd.read_csv("titanic.csv")
# print(df.columns)

df.drop(['PassengerId','Name','Ticket','SibSp','Cabin','Parch','Embarked'],axis='columns',inplace=True)

# print(df.columns)

target=df.Survived
inputs=df.drop('Survived',axis='columns')

dummies=pd.get_dummies(inputs.Sex)

inputs=pd.concat([dummies,inputs],axis='columns')

# print(inputs.head())
inputs=inputs.drop(['Sex'],axis='columns')

# print(inputs.head())

# print(inputs.columns[inputs.isna().any()])

inputs.Age=inputs.Age.fillna(inputs.Age.mean())

# print(inputs.columns[inputs.isna().any()])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(inputs,target,test_size=0.2,random_state=0)

# print(X_train.shape,Y_train.shape)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,Y_train)
# print(model.score(X_test,Y_test))

# print(Y_test[:10])
print(model.predict_proba(X_test[:10]))