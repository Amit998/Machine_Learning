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

# print(df[df['twitts'].str.contains('gmail.com')])

# print(df.iloc[3713]['twitts'])

import re

# x="@securerecs arghh Me Please  markbradbury_16@hotmail.com"

# print(re.findall(r'([a-z0-9._+@]+[a-zA-Z]+\.[a-zA-Z]+\b)',x))

# df['emails']=df['twitts'].apply(lambda x: re.findall(r'([a-z0-9._]+[a-zA-Z0-9]+\.[a-zA-Z0-9]+\b)',x))
# df['twitts']=df['twitts'].apply(lambda x: re.sub(r'([a-z0-9._+]+[a-zA-Z0-9]+\.[a-zA-Z0-9]+\b)','',x))
# 
# df['email_count']=df['emails'].apply(lambda x: len(x))



# print(df[df['email_count'] > 0].sample(10))
#  print(df.sample(100))

# Count URLs and Remove it


# x='hello https://stackoverflow.com/questions/45913520/python-extract-urls-from-a-text-file-with-no-html-tags'
# print(re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',x))

# df['website_links']=df['twitts'].apply(lambda x: re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',x))

# print(re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',x))

# df['website_count']=df['website_links'].apply(lambda x: len(x))

# df['link_flags']=df['twitts'].apply(lambda x: re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',x))


# df['twitts']=df['twitts'].apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',x))



# print(   df[df['website_count'] > 0 ].sample(10))




# print(df['twitts'].str.contains('rt').sample(10))


# x='rt @username: hello world'

# print(re.sub(r'\brt\b','',x).strip())

# df['twitts']=df['twitts'].apply(lambda x: re.sub(r'\brt\b','',x))

# print(df.sample(10))




##### Special Chars Removal or Punctuation removal

# x='@JackAllTimeLow cant believe its endingg .. wh...  ...'
# print(re.sub(r'[\w]+',"",x))
# print(re.sub(r'[^\w]+'," ",x))

# df['twitts']=df['twitts'].apply(lambda x: re.sub(r'[^\w]+'," ",x))


## Remove Spaces

x="Hi      Hello     how are    You"


# print(x)
# x=' '.join(x.split())
# print(x)


# df['twitts']=df['twitts'].apply(lambda x: ' '.join(x.split()))


#### Remove Bewutifulsou4 

x ='<html><h1>Thanks For Listing To Me</h1></html>'

from bs4 import BeautifulSoup


# x.replace('<html><h1>','')
# print(x)

# print(BeautifulSoup(x,'lxml').getText().strip())

# print(x)

# df['twitts']=df['twitts'].apply(lambda x: BeautifulSoup(x,'lxml').getText().strip())

# print(df.sample(10))


x= 'àèìòùÀÈÌÒÙ'

import unicodedata


def Remove_accented_Char(x):
    x=unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore')
    return x



# df['twitts']=df['twitts'].apply(lambda x: Remove_accented_Char(x))


### remove Stop Words


# x ='this is stop words'

# print(' '.join([t for t in x.split() if t not in stop_words]))


# df['twitts_no_stop_word']=df['twitts'].apply(lambda x: ' '.join([t for t in x.split() if t not in stop_words]))

# print(df.sample(10))

# print(Remove_accented_Char(x))



# Convert into base or root

import spacy


nlp=spacy.load('en_core_web_sm')

x='chocolates balls times'


def make_to_base(x):
    x_list=[]
    doc=nlp(x)
    for token in doc:
        lemma=str(token.lemma_)
        if lemma == '-PRON-' or lemma == 'be':
            x_list.append(lemma)
    return ' '.join(x_list)

# df['twitts']=df['twitts'].apply(lambda x: make_to_base(x))


# print(df.sample(10))

# print(make_to_base(x))


## common word removal

x="this is this okay bye"

text=' '.join(df['twitts'])
# print(len(text))

text=text.split()

# print(len(text))

freq_comm=pd.Series(text).value_counts()

# f20=freq_comm[0:20]
# print(f20)


# df['twitts']=df['twitts'].apply( lambda x:  ' '.join([ t for t in x if t not in f20 ])  )

# text=' '.join(df['twitts'])
# text=text.split()
# freq_comm=pd.Series(text).value_counts()
# f20=freq_comm[0:20]
# print(f20)
# print(freq_comm)


# Rare Words


print(freq_comm.tail())

Rare20=freq_comm[-20:]
# print(Rare20)


# df['twitts']=df['twitts'].apply( lambda x:  ' '.join([ t for t in x if t not in Rare20 ])  )


# print(freq_comm.tail())

## Word Cloud Visualization


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# # %matplotlib inline
# text=' '.join(df['twitts'])
# print(len(text))

# wc=WordCloud(width=800,height=400).generate(text)
# plt.imshow(wc)
# plt.axis('off')
# plt.show()


### Spelling Correction

from textblob import TextBlob
# x='Thanks for watchign it'

# x=TextBlob(x).correct()
# print(x)


## Tokenization


x="thanks#watching this. video"

# print(TextBlob(x).words)

# doc =nlp(x)

# for token in doc:
    # print(token)



## Detecting Noun


x="To type a lowercase character by using a key combination that includes the SHIFT key, hold down the CTRL+SHIFT+symbol keys simultaneously, and then release them before you type the letter. For example, to type a ô, hold down CTRL, SHIFT and ^, release and type o."


# doc=nlp(x)


# for noun in doc.noun_chunks:
    # print(noun)



# language Translation and Detaction


tb=TextBlob(x)

# print(tb.detect_language())

# print(tb.translate(to='bn'))


## Sentiment Classifire

from textblob.sentiments import NaiveBayesAnalyzer

z="we all stands together. we are gonna win this fight"

demo=TextBlob(x,analyzer=NaiveBayesAnalyzer())
print(demo.sentiment())
