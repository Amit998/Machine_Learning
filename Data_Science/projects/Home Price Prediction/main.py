from matplotlib.pyplot import axis
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
# print(df6.price_per_sqft.describe())
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

# print(df7.head())


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
    
# print(df7.columns)
# plot_scatter_chart(df7,"Sarjapur") 


def remove_bhk_outliner(df):
    bhk_stats={}
    exclude_indices=np.array([])
    for location , location_df in df.groupby('location'):
        
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }

        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if (stats and  stats['count']>5):
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    # print(bhk_stats)
    

    return df.drop(exclude_indices,axis='index')


df8=remove_bhk_outliner(df7)
# print(df8.shape)

# plot_scatter_chart(df7,"Sarjapur") 
# plot_scatter_chart(df8,"Sarjapur") 
# print(df8)


import matplotlib
# matplotlib.rcParams["figure.figsize"]=(10,5)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel("Price Per Squre Fit")
# plt.ylabel("Count")
# plt.show()


# print(df8[df8.bath>10].count())

# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("Number of bathrooms")
# plt.ylabel("Count")
# plt.show()


# print(df8[df8.bath>df8.bhk+2])


df9=df8[df8.bath<df8.bhk+2]


# print(df9.columns)
df10=df9.drop(['size'],axis='columns')



dummy=pd.get_dummies(df10.location)
# print(df10.head())

df11=pd.concat([df10,dummy.drop(['other'],axis='columns')],axis='columns')
df12=df11.drop(['location','price_per_sqft'],axis='columns')
# print(df12.columns)
# print(df13.head())
x=df12.drop('price',axis='columns')
y=df12.price

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

# print(x_train.shape,y_train.shape)

from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)

score=lr_clf.score(x_test,y_test)
# print(score)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

# print(cross_val_score(LinearRegression(),x,y,cv=cv))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model_using_gridsearchcv(x,y):
    algos={
        'Linear_Regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'normalize':['random','cyclic']
            }
        },
        'Decision_Tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append(
            {
                'model':algo_name,
                'best_score':gs.best_score_,
                'best_params':gs.best_params_
            }
        )
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# ans=find_best_model_using_gridsearchcv(x,y)
# print(ans)




def predict_price(x,location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    x=np.zeros(len(x.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if (loc_index >= 0):
        x[loc_index]=1
    
    return lr_clf.predict([x])[0]

# print(np.where(x.columns=='Yelachenahalli'))

# print(x.columns)

# print(predict_price(x,'1st Phase JP Nagar',1000,3,3))

import pickle
with open('bangalore_home_price_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns={
    'data_columns':[col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))