from os import name
import numpy as np
import pandas as pd
import tensorflow as tf
import ktrain
from ktrain import text




# data_train=pd.read_excel('train.xlsx',sheet_name=None,engine='openpyxl')
# data_train=pd.read_excel('test.xlsx',sheet_name=None,engine='openpyxl')

data_train = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx',  dtype = str,engine='openpyxl')
data_test = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx',  dtype = str,engine='openpyxl')


# print(data_test.shape)
# print(data_train.shape)
# print(data_train.head)

(x_train,y_train),(x_test,y_test),preprocess=text.texts_from_df(
    train_df=data_train,
    text_column='Reviews',
    label_columns='Sentiment',
    val_df=data_test,
    maxlen=500,
    preprocess_mode='bert',
    
)


# print(x_train[0].shape)

model=text.text_classifier(
    name='bert',
    train_data=(x_train,y_train),
    preproc=preprocess
)

# #get learning rate

learner=ktrain.get_learner(
    model=model,
    train_data=(x_train,y_train),
    val_data=(x_test,y_test),
    batch_size=6,

)


# #this might take days to run
# # learner.lr_find()
# # learner.lr_plot()

# learner.fit_onecycle(lr=2e-5,epochs=1)


# predictor=ktrain.get_predictor(learner.model,preprocess)

# data=["This movie sucks, Plot was worst",'I want my money back','this hi awsome move']

# predictor.predict(data)

# predictor.save('/contents/bert')