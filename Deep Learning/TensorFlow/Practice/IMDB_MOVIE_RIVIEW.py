
import os
from tensorflow.python.keras import initializers, models
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.ops.gen_array_ops import shape, split
from tensorflow.python.ops.variables import trainable_variables
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers

import matplotlib.pyplot
physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)
import  tensorflow_datasets as tfds




(ds_train,ds_test),ds_info=tfds.load(
    "imdb_reviews",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# print(ds_train)
# print(ds_info)

#tokenization

tokenizer=tfds.deprecated.text.Tokenizer()

def build_vocabulary():
    vocabulary=set()
    for text,_ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))
    
    return vocabulary

vocabulary=build_vocabulary()

# print(vocabulary)

encoder=tfds.deprecated.text.TokenTextEncoder(
    vocabulary,
    oov_token="<UNK>",
    lowercase=True,
    tokenizer=tokenizer
)

def my_encoding(text_tensor,label):
    return encoder.encode(text_tensor.numpy()),label

def encode_map(text,label):
    encoded_text,label=tf.py_function(
        my_encoding,inp=[text,label],Tout=(tf.int64,tf.int64)
    )

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text,label

AUTOTUNE=tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(encode_map,num_parallel_calls=AUTOTUNE).cache()
ds_train=ds_train.shuffle(10000)
ds_train=ds_train.padded_batch(32,padded_shapes=([None],()))
ds_train=ds_train.prefetch(AUTOTUNE)

ds_test=ds_test.map(encode_map)
ds_test=ds_test.padded_batch(32,padded_shapes=([None],()))


model=keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary)+2,output_dim=32),
        layers.GlobalAveragePooling1D(),
        # layers.LSTM(activation='tanh')
        layers.Dense(64,activation='relu'),
        layers.Dense(1),

    ]
)


model.compile(
    
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4,clipnorm=1),
    metrics=["accuracy"],
)


model.fit(ds_train,epochs=10,verbose=2)
model.save("IMDB_REVIEW_PREDICTOR")
print(model.evaluate(ds_test,verbose=2))