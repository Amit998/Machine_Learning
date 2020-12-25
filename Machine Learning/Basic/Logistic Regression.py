import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance_data.csv")


# plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
# plt.show()



X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


model.fit(X_train, y_train)


y_predicted = model.predict(X_test)



print(model.predict_proba(X_test))

model.score(X_test,y_test)




import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))



def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y

age = 35
print(prediction_function(age))

age = 43
print(prediction_function(age))