import spacy
nlp=spacy.load("en_core_web_sm")


doc=nlp(u'The Solar Power industry continues to grow as demand \ for solarpower increases. Solar-Power cars  are gaining popularity.')

doc=nlp(u"Let's visit st. louis in the U.S next year.")


for t in doc:
    print(t)


import nltk
from nltk.stem.porter import *

p_stemmer=PorterStemmer()
words=['run','runner','running','ran','runs','easily','easier','fairly']
words=['consolingly']
for word in words:
    print(word+' --> '+p_stemmer.stem(word))


from nltk.stem.snowball import SnowballStemmer 
s_stemmer=SnowballStemmer(language='english')

words=['consolingly']

for word in words:
    print(word+' --> '+s_stemmer.stem(word))

pharse='I am meeting him tommrow at the meeting'
for word in pharse.split():
    print(word+'" Snowball "--> '+s_stemmer.stem(word))
    print(word+'" Porter " --> '+p_stemmer.stem(word))