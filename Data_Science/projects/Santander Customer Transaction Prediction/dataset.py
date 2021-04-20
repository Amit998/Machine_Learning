import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import TensorDataset
import math
def get_data():
    train_data=pd.read_csv("D:/study/datasets/CSV_data/Santander Customer Transaction Prediction/train.csv")
    y=train_data["target"]
    x=train_data.drop(["ID_code"],axis=1)
    x_tensor=torch.tensor(x.values,dtype=torch.float32)
    y_tensor=torch.tensor(y.values,dtype=torch.float32)

    ds=TensorDataset(x_tensor,y_tensor)
    
    train_ds,val_ds=random_split(ds,[int(0.8*len(ds)),math.ceil(0.2*len(ds))])

    test_data=pd.read_csv("D:/study/datasets/CSV_data/Santander Customer Transaction Prediction/test.csv")
    test_ids=test_data["ID_code"]
    x=test_data.drop(["ID_code"],axis=1)
    x_tensor=torch.tensor(x.values,dtype=torch.float32)
    y_tensor=torch.tensor(y.values,dtype=torch.float32)

    test_ds=TensorDataset(x_tensor,y_tensor)
    
    # test_ds,val_ds=random_split(test_ds,[int(0.8*len(test_ds)),int(0.2*len(test_ds))])

    return train_ds,val_ds,test_ds,test_ids
