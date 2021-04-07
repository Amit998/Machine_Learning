import os
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from tensorflow.python.ops.gen_array_ops import split

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import pickle


def filter_train(line):
    split_line=tf.strings.split(line,",",maxsplit=4)
    dataset_belonging=split_line[1] #train test
    sentiment_category=split_line[2] # pos,neg,unsup

    return (
        True
        if dataset_belonging == 'train' and sentiment_category != 'unsup'
        else False
    )

def filter_test(line):
    split_line=tf.strings.split(line,",",maxsplit=4)
    dataset_belonging=split_line[1] #train test
    sentiment_category=split_line[2] # pos,neg,unsup

    return (
        True
        if dataset_belonging == 'test' and sentiment_category == 'unsup'
        else False
    )

file_path="D:/study/datasets/CSV_data/imdb.csv"

# ds_train=tf.data.TextLineDataset(file_path)

# for line in ds_train.skip(1).take(5):
    # print(tf.strings.split(line,",",maxsplit=4))


ds_train=tf.data.TextLineDataset(file_path).filter(filter_train)
ds_test=tf.data.TextLineDataset(file_path).filter(filter_test)

# TODO

# 1. CREATE VOCUBULARY
# 2. NUMERICALIZE TEXT STR-> indicies (TokenTextEncoder)
# 3. pad th batches so we can sent in to an RNN for example

tokenizer=tfds.deprecated.text.Tokenizer()

def build_vocabulary(ds_train,threshold=200):
    frequency={}
    vocabulary=set()
    vocabulary.update(["sootoken"])
    vocabulary.update(["eostoken"])

    for line in ds_train.skip(1):
        split_line=tf.strings.split(line,",",maxsplit=4)
        review=split_line[4]
        tokenize_text=tokenizer.tokenize(review.numpy().lower())

        for word in tokenize_text:
            if word not in frequency:
                frequency[word]=1
            else:
                frequency[word]+=1
            if frequency[word]== threshold:
                vocabulary.update(tokenize_text)
    return vocabulary

# vocabulary=build_vocabulary(ds_train)
# vocab_file=open("vocabulary.obj","wb")
# pickle.dump(vocabulary,vocab_file)

vocab_file=open("vocabulary.obj","rb")
vocabulary=pickle.load(vocab_file)

encoder=tfds.deprecated.text.TokenTextEncoder(
    list(vocabulary),
    oov_token="<UNK>",
    lowercase=True,
    tokenizer=tokenizer

)

def my_encoder(text_tensor,label):
    encoder_text=encoder.encode(text_tensor.numpy())
    return encoder_text,label

def encode_map_fn(line):
    split_line=tf.strings.split(line,",",maxsplit=4)
    label_str=split_line[2]
    review="sostoken"+split_line[4] + "eostoken"

    label = 1 if label_str == "pos" else  0

    (encoded_text,label)=tf.py_function(
        my_encoder,inp=[review,label],Tout=(tf.int64,tf.int32),
    )

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text,label

AUTOTUNE=tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(encode_map_fn,num_parallel_calls=AUTOTUNE).cache()
ds_train=ds_train.shuffle(25000)
ds_train=ds_train.padded_batch(32,padded_shapes=([None],()))


ds_test=ds_test.map(encode_map_fn)
ds_test=ds_test.padded_batch(32,padded_shapes=([None],()))


model=keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary)+2,output_dim=32),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64,activation='relu'),
        layers.Dense(1),
    ]
)
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),optimizer=keras.optimizers.Adam(lr=3e-4,clipnorm=1),metrics=["accuracy"],
)

model.fit(ds_train,epochs=15,verbose=2)

print(model.evaluate(ds_test))
# model.save('model.h4') 