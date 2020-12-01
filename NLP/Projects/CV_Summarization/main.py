import spacy
import pickle
import random

from word2number import w2n


lst=['one','two','three']


for i in lst:
    print(w2n.word_to_num(i))


df["columen_name"]=df["columen_name"].apply( lambda x: w2n.word_to_num(x) )