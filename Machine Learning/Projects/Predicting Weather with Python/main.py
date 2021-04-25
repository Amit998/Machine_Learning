PATH="D:/study/datasets/CSV_data/Rain/weatherAUS.csv"

import pandas as pd
from  neuralprophet import NeuralProphet, forecaster
from matplotlib import pyplot as plt
import pickle

df=pd.read_csv(PATH)

# print(df.head(10))

# print(df.columns)
# print(df.Location.unique())
# print(df.dtypes)

# melb=df[df['Location']=='Melbourne']
# melb['Date']=pd.to_datetime(melb['Date'])
# # print(melb.dtypes)


# melb['Year']=melb['Date'].apply(lambda x:x.year)
# melb=melb[melb['Year']<=2015]


# # plt.plot(melb['Date'],melb['Temp3pm'])
# # plt.show()


# data = melb[['Date', 'Temp3pm']] 
# data.dropna(inplace=True)
# data.columns = ['ds', 'y'] 
# print(data.head())

# m=NeuralProphet()
# m.fit(data,freq='D',epochs=1000)

# # #FORCASE AWAY

# future = m.make_future_dataframe(data, periods=900)
# forecast = m.predict(future)
# print(forecast.tail())

# # plot1=m.plot(forecast)
# # plot2=m.plot_components(forecast)


# with open("forecast_model.pkl","wb") as f:
#     pickle.dump(m,f)

# with open("forecast_model.pkl","rb") as f:
#     m=pickle.load(f)


# future=m.make_future_dataframe(data,periods=12000)
# forecast=m.predict(future)
# forecast.head()

import pandas as pd
from prophet import Prophet