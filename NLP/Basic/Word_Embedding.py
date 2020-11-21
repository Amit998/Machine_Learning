from tensorflow.keras.preprocessing.text import one_hot

sentences=[
    'i am amit',
    'This is glass of milk',
    'this is glass of juice',
    'The Cup of tea',
    'i am a good developer',
    'I know Flutter',
    'Hire Me!',
    'my name is amit',
]

voc_size=10000


oneHot_repr=[one_hot(words,voc_size)for words in sentences]
# print(oneHot_repr)

from tensorflow.keras.layers import Embedding
from  tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

import pandas as pd

len_list=[ len((data).split(' ')) for data in sentences ]
# print(len_list)
# max_lenth=max(len_list)
max_lenth=8

embedded_doc=pad_sequences(oneHot_repr,padding='pre',maxlen=max_lenth)

# print(embedded_doc)

dim=10

model=Sequential()

model.add(Embedding(voc_size,10,input_length=max_lenth))
model.compile('adam','mse')

# print(model.summary())

# print(model.predict(embedded_doc))


# print(model.summary())

print(embedded_doc[0])
print(model.predict(embedded_doc[0]))