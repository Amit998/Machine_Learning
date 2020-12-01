import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(10,5)


df1=pd.read_csv("dataset/Bengaluru_House_Data.csv")
# print(df1.head())
print(df1.columns)
# print(df1.groupby('area_type')['area_type'].agg('count'))
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
# print(df2.head())
# print(df2.isnull().sum())
df3=df2.dropna()
# print(df3.head())
# print(df3['size'].unique())
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]) )
# print(df3[df3.bhk > 20])
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
def convert_sqft_to_num(x):
    token=x.split('-')
    if (len(token) == 2):
        return (float(token[0]) + float(token[1]))/2
    try:
        return float(x)
    except:
        return None
# print(convert_sqft_to_num("10 - 2 "))
df4=df3.copy()
df4['total_sqft']=df3['total_sqft'].apply( lambda x: convert_sqft_to_num(x))
# print((df3[~df3['total_sqft'].apply(is_float)]))
# print((df4[~df4['total_sqft'].apply(is_float)]))
df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
# print(df5.head())
# print(df5.location.unique())
df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats)
location_stats_less_then_10=location_stats[location_stats<=10]

# print(len(df5.location.unique()))
df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_then_10 else x)


location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats)
# print(df5.location.unique())



# print(df5[df5.total_sqft/df5.bhk<300].shape)

df6=df5[~(df5.total_sqft/df5.bhk<300)]

# print(df6.columns)
print(df6.price_per_sqft.describe())
# print(df6.location.describe())


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduce_df=subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduce_df],ignore_index=True)

    return df_out

df7=remove_pps_outliers(df6)

print(df7.head())


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (10,5)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
    
plot_scatter_chart(df7,"Rajaji Nagar") 