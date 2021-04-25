import warnings
from pandas._libs import interval
warnings.simplefilter("ignore")


import pandas as pd

from fbprophet import Prophet
# from  prophet import Prophet


df=pd.read_csv("dataset.csv")

# print(df.head())

# print(df.describe())

# print(df['Product'].unique())
# print(df['Store'].unique())

# print(df.dtypes)
df['Year']=df['Time Date'].apply(lambda x:str(x)[-4:])
df['Month']=df['Time Date'].apply(lambda x:str(x)[-6:-4])
df['Day']=df['Time Date'].apply(lambda x:str(x)[:-6])

df['ds']=pd.DatetimeIndex(df['Year']+'-'+df['Month']+'-'+df['Day'])

# # print(df.tail())

df.drop(['Time Date','Product','Store','Month','Day','Year'],axis=1,inplace=True)

# # print(df.tail())
# ##For particular product
# # df[df['Product']=='xyz']


data=df[['ds','Value']]


data.columns=['ds','y']
print(data.head())


m = Prophet(interval_width=0.95, daily_seasonality=True)
m.fit(data)
