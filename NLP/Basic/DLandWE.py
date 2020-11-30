from scipy.sparse.construct import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D 
import numpy as np
from numpy import array
import pandas as pd
from sklearn.model_selection import train_test_split


data=pd.read_csv('twitter30k_cleaned.csv')
# print(data['sentiment'].value_counts())
text=data['twitts'].tolist()
# print(text)
y=data['sentiment']
token=Tokenizer()
token.fit_on_texts(text)



# print(token.word_index)
vocab_size=len(token.word_index) +1
# print(vocav)
x=['i to the a and']
# print(token.texts_to_sequences(x))
encoded_text=token.texts_to_sequences(text)
max_length=120
x=pad_sequences(encoded_text,maxlen=max_length,padding='post')
# print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
# print(x_train.shape)
vec_size=300


model=Sequential()
model.add(Embedding(
    vocab_size,
    vec_size,
    input_length=max_length
))

model.add(Conv1D(64,8,activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

y_pred=model.predict(x_test)


temp_pred=[]
for pred in y_pred:
    if(pred[0] > 0.5):
        temp_pred.append(1)
    else:
        temp_pred.append(0)

# print(y_test)
# print(y_pred)

for i,j in zip(temp_pred,y_test):
    if(i==j):
        print('True')
    else:
        print('false')


def get_encoded(x):
    x=token.texts_to_sequences(x)
    x=pad_sequences(x,maxlen=max_length,padding='post')
    return x


x=['worst service, Will Not Come Again']

print(model.predict(get_encoded(x)))

