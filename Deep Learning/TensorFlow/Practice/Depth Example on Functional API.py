import os
from tensorflow.python.keras.engine.training import Model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.datasets import mnist

physical_divice=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_divice[0],True)


# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001


train_df = pd.read_csv("D:/study/datasets/multidigit/train.csv")
test_df = pd.read_csv("D:/study/datasets/multidigit/test.csv")
train_images = "D:/study/datasets/multidigit/train_images/" + train_df.iloc[:, 0].values
test_images = "D:/study/datasets/multidigit/test_images/" + test_df.iloc[:, 0].values

train_labels = train_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 1:].values

