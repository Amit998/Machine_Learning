import pandas as pd
import numpy as np
import spacy

from spacy.lang.en.stop_words import STOP_WORDS as stop_words

df=pd.read_csv('twitter4000.csv',encoding='latin1')
# print(df)
# print(df['sentiment'].value_counts())

# Word Counts



df['word_counts']=df['twitts'].apply( lambda x: len(str(x).split()))
# print(df.sample(5))
# print(df['word_counts'].max())

# print(df[df['word_counts']==1])

# Characters Count




def char_counts(x):
    s=x.split()
    x=''.join(s)
    return len(x)

df['char_counts']=df['twitts'].apply(lambda x: char_counts(x))
# print(df.sample(5))



### Average Word Length


def avg_word_count(x):

    word=len(x.split())
    char=x.split()
    x=''.join(char)
    charlen=len(x)
    return charlen//word



df['avg_char_in _word_len']=df['char_counts']//df['word_counts']
# print(df.head())

# df['avg_char_in _word']=df['twitts'].apply(lambda x: avg_word_count(x))
# print(df.head())


# Stop Words Count

# print(len(stop_words))


x='this is the text data'
lst=[]

# [lst.append(t) for t in x.split() if t in stop_words]
# print(lst)

# df['stop_words_length']=df['twitts'].apply(lambda x: len([lst.append(t) for t in x.split() if t in stop_words]))

# df['stop_words_length']=df['twitts'].apply(lambda x: len([t for t in x.split() if t in stop_words]))


# print(df.head())


### Count #HashTag and @Mentions


x='this is #hastag and this is @Mentions'

# x.split('')

# [t for t in x.split() if t.startswith("#")] 


# df['hastag_len']=df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith("#")]) )
# df['metions_len']=df['twitts'].apply(lambda x: len([t for t in x.split() if t.startswith("@")]) )


# print(df.head())

# print(t)



## if numric digits

# x='This is 1 and 2'
# len([ t for t in x.split() if t.isdigit() ])
# print(lst)

# df['neumeric_count']=df['twitts'].apply(lambda x: len([ t for t in x.split() if t.isdigit() ]) )
# print(df.sample(10))


## Upper Case And Lower Case Word Count


# x='I AM HAPPY . are you happy?'


# print(len([t for t in x.split() if t.isupper()]))


# df['upper_value_count']=df['twitts'].apply(lambda x: len([t for t in x.split() if t.isupper()]) )
# df['lower_value_count']=df['twitts'].apply(lambda x: len([t for t in x.split() if not t.isupper()]) )


#Lower case conversion

df['twitts']=df['twitts'].apply(lambda x: str(x).lower())


print(df.sample(10))