import spacy
import nltk
nlp=spacy.load("en_core_web_sm")
nltk.download('punkt')
from  nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
text = " 1234 I’m designing a document and don’t want to get bogged down in what the text actually says $."
tokens=word_tokenize(text)
import string
import re
# print(tokens)

tokens=[ word.lower() for word in tokens ]
# print(tokens)


re_punc=re.compile('[%s]' % re.escape(string.punctuation))
# print(re_punc)

stripped=[re_punc.sub('',w) for w in tokens]
# print(stripped)

words=[word for word in stripped if word.isalpha()]
print(words)


stop_words=set(stopwords.words('english'))
words=[w for w in words if not w in stop_words]
print(words)

# doc = nlp(text)
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(len(stopwords.words('english')))
# print(len(nlp.Defaults.stop_words))

# print(nlp.vocab['mystery'].is_stop)
# print(nlp.vocab['me'].is_stop)

# nlp.Defaults.stop_words.add('mystery')
# nlp.vocab['mystery'].is_stop=True

# print(nlp.vocab['mystery'].is_stop)

# nlp.Defaults.stop_words.remove('me')
# nlp.vocab['me'].is_stop=False

# print(nlp.vocab['me'].is_stop)

