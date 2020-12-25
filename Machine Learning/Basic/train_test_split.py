import pandas as pd
import matplotlib.pyplot as plt






df = pd.read_csv("carprices.csv")
# plt.scatter(df['Mileage'],df['Sell_Price'])
# plt.scatter(df['Age(yrs)'],df['Sell_Price'])
# plt.show()

X = df[['Mileage','Age(yrs)']]
y = df['Sell_Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)

print(clf.predict(X_test))