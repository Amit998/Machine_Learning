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
from wordcloud import wordcloud



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Embedding,GRU,LSTM,RNN,SpatialDropout1D

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score





dir="D:/study/datasets/CSV_data/Fake-News-Classifier-master/data"