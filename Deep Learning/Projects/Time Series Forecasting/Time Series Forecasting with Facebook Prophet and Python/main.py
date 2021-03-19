import pandas as pd
# import fbprophet
import matplotlib.pyplot as plt
# %matplotlib inline
# import pysta


df=pd.read_csv('monthly-milk-production-pounds.csv')
# print(df.head())
# print(df.tail())

# df.dropna()
# print(df.tail())

# df.plot()
# plt.show()

df.columns=['ds','y']

