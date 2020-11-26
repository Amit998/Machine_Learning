from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D 
import numpy as np
from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split





data=pd.read_csv('training1600000.csv')


newData=data[0:4000]
# print(len(newData))

print(newData)