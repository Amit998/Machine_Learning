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
from tensorflow.python.keras.backend import learning_phase_scope


data=pd.read_csv('twitter4000.csv')


contractions = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


import re


# print(data)

text=' '.join(data['twitts'])
text=text.split()
freq_comm=pd.Series(text).value_counts()
rare=freq_comm[freq_comm.values == 1]
# print(rare)
# print(freq_comm)



def get_clean_text(x):
    if (type(x) is str):
        x=x.lower()
        for key in contractions:
            value=contractions[key]
            x=x.replace(key,value)
        x=re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9+._-]+\.[a-zA-Z0-9+._-]+)','',x)
        x = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)
        x = re.sub('RT', "", x)
        x = re.sub('[^A-Z a-z]+', '', x)
        x=' '.join([t for t in x.split() if t not in rare])
        return x
    else:
        return x


data['twitts']=data["twitts"].apply(lambda x: get_clean_text(x))
# print(data['twitts'])



text=data["twitts"].tolist()
# print(text)
y=data["sentiment"]
token=Tokenizer()
token.fit_on_texts(text)
# print(token)
vocab_size=len(token.word_index) + 1
# print(token.word_index)
# print(vocab_size)

encoded_text=token.texts_to_sequences(text)
max_len=120
x=pad_sequences(encoded_text,maxlen=max_len,padding='post',value=0)

# print(x)


# GloVe Vector


glove_vec={}

file=open('D:/download/archive (5)/glove.twitter.27B.200d.txt',encoding='utf-8')


for line in file:
    values=line.split()
    word = values[0]
    vectors=np.asarray(values[1:])
    glove_vec[word]=vectors
file.close()

# print(len(glove_vec.keys()))

# print(glove_vec.get('You'))

word_vec_matrix=np.zeros((vocab_size,200))


# print(word_vec_matrix)
# print(word_vec_matrix.shape)

for word,index in token.word_index.items():
    vector=glove_vec.get(word)
    # print(vector.shape)
    if(vector is not None):
        word_vec_matrix[index]=vector
    else:
        pass
# print(vector)


##TF2.0 keras model building

# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=40,test_size=0.2)

# print(x_train[:5],'\n',x_test[:5])
# print('\n')
# print(y_train[:5],'\n',y_test[:5])

vec_size=200



model=Sequential()
model.add(Embedding(
    vocab_size,
    vec_size,
    input_length=max_len,
    weights=[word_vec_matrix],
    trainable=False
))

model.add(Conv1D(64,8,activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))



y_pred=model.predict(x_test)


temp_pred=[]
for pred in y_pred:
    if(pred[0] > 0.5):
        temp_pred.append(1)
    else:
        temp_pred.append(0)

# print(y_test,y_pred)

for i,j in zip(temp_pred,y_test):
    if(i==j):
        print('True')
    else:
        print('false')



def get_encoded(x):
    x=get_clean_text(x)
    x=token.texts_to_sequences(x)
    x=pad_sequences(x,maxlen=max_len,padding='post')
    return x


x=['worst service, Will Not Come Again']

print(model.predict(get_encoded(x)))



