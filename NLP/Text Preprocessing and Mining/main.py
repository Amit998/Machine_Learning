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

# df['twitts']=df['twitts'].apply(lambda x: str(x).lower())


# print(df.sample(10))

# Contraction to Expansion


cList = {
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
  "you've": "you have",
  "i'm":"i am"
}
# print(len(cList))
x=" i'm don't he'll "


def cont_to_exp(x):
    if type(x) is str:
        for key in cList:
            value=cList[key]
            x=x.replace(key,value)
        return x
    else:
        return x

x=x.split()
# print(cont_to_exp(x))

lst=[]
# print([(cont_to_exp(w)) for w in x])


# print(df.sample(10))
# df['twitts']=df['twitts'].apply(lambda x: cont_to_exp(x))
# print(df.sample(10))


# Count and Email Removal

print(df[df['twitts'].str.contains('gmail.com')])

print(df.iloc[2448]['twitts'])