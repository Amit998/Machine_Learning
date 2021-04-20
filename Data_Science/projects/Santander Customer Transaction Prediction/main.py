import pandas as pd


train=pd.read_csv("D:/study/datasets/CSV_data/Santander Customer Transaction Prediction/train.csv")

# print(train.head(10))
print(train.corr().abs())