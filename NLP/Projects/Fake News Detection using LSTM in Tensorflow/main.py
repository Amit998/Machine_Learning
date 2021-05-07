import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from string import punctuation
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import seaborn as sns
import nltk
from wordcloud import WordCloud



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Embedding,GRU,LSTM,RNN,SpatialDropout1D


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

# Index(['title', 'text', 'subject', 'date'], dtype='object')




dir="D:/study/datasets/CSV_data/Fake-News-Classifier-master/data"

fake=pd.read_csv(f"{dir}/Fake.csv")
real=pd.read_csv(f"{dir}/True.csv")

real=real.drop(8970,axis=0)


# print(fake.columns)

# print(fake['subject'].value_counts())

# plt.figure(figsize=(10,6))
# sns.countplot('subject',data=fake)
# plt.show()

#WORD COLUD


# text=''.join(real['text'].tolist())
# plt.figure(figsize=(10,16))
# wordcloud=WordCloud(width=1920,height=1000).generate(text)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()


# print(real.sample(5))

unknown_publishers=[]

for index,row in enumerate(real.text.values):
    try:
        record=row.split(' - ',maxsplit=1)

        assert (len(record[0])<120)
    except:
        # print(e)
        unknown_publishers.append(index)


publisher=[]
tmp_text=[]


for index,row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('Unknown')
    else:
        record=row.split(' - ',maxsplit=1)
        publisher.append(record[0])
        tmp_text.append(record[1])

real['publisher']=publisher
real['text']=tmp_text

# print(real.shape)
# print(real.head(10))
# print(real.columns)

empty_fake_index=[index for index,text in enumerate(fake.text.tolist()) if str(text).split()==""  ]

print(empty_fake_index)
