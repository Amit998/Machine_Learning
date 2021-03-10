import tensorflow as tf
import string
import requests


response=requests.get('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')

# print(response.text)

data = response.text.split('\n')

data=data[253:]


# print(data[0])

data=" ".join(data)
# print(data)

def clean_text(doc):
    tokens=doc.split()
    table=str.maketrans('','',string.punctuation)
    tokens=[w.translate(table) for w in tokens]
    # print(tokens)
    tokens=[word for word in tokens if word.isalpha()]
    tokens=[word.lower() for word in tokens]

    return tokens

tokens=clean_text(data)
# print(tokens[:10])
# print(len(tokens))



# print(len(set(tokens)))


length=50+1


lines=[]

for i in range(length,len(tokens)):
    seq=tokens[i-length:i]
    # print(len(seq),i-length,i)

    line=' '.join(seq)
    lines.append(line)

    if(i > 200000):
        break

# print(len(lines))

# print(lines[0])

# print(tokens[0])

##BUILD LSTM Model and prepare x and y



import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer=Tokenizer()
tokenizer.fit_on_texts(lines)
sequences=tokenizer.texts_to_sequences(lines)


sequences=np.array(sequences)


x,y=sequences[:,:-1], sequences[:,-1]
vocab_size=len(tokenizer.word_index) + 1

y=to_categorical(y,num_classes=vocab_size)

seq_length=x.shape[1]



##LSTM

model=Sequential()
model.add(Embedding(vocab_size,50,input_length=seq_length))

model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100,activation='relu'))

model.add(Dense(vocab_size,activation='softmax'))

# print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,batch_size=256,epochs=10)

import pickle

pickle_out=open('classifier.pkl','wb')
pickle.dump(model,pickle_out)
pickle_out.close()

# print(x_test)
# seed_text=lines[123456]

# def generate_text_seq(model,tokenizer,text_seq_length,seed_text,n_word):
#     text=[]

#     for _ in range(n_word):
#         encoded=tokenizer.texts_to_sequences([seed_text])[0]
#         encoded=pad_sequences([encoded],max_len=text_seq_length,truncating='pre')

#         y_pred=model.predict_classes(encoded)


#         predicted_word=''
#         for word,index in tokenizer.word_index.items():
#             if (index == y_pred):
#                 predicted_word= word
#                 break
#         seed_text=seed_text+' '+ predicted_word
#         text.append(predicted_word)

#         return ''.join(text)

# print(seed_text)
# seeded_text=generate_text_seq(model,tokenizer,seq_length,seed_text,100)
# print(seeded_text)