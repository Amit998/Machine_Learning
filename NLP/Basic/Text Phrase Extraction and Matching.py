from numpy.core import machar
import spacy
from spacy.matcher import Matcher, matcher
from spacy.tokens import Span
from spacy import displacy


nlp=spacy.load('en_core_web_sm')

doc=nlp('Hello Word!')

# print(doc)

# for token in doc:
#     print(token)

pattern=[
    {
        "LOWER":"hello"
    },
    {
        "IS_PUNCT":True,
        'OP':'?'
    },
    {
        "LOWER":"world"
    }
]

# matcher=Matcher(nlp.vocab)
# matcher.add('Word',None,pattern)
# doc=nlp("hello World !")
# matches=matcher(doc)

# print(matches)

# for match_id,start,end in matches:
#     string_id=nlp.vocab.strings[match_id]
#     span=doc[start:end]
#     print(match_id,start,end,span.text)


text='My Phone Number is 123456. Oh its Wrong'


import re

# print(re.search(r'\d{4}',text))


# print(re.findall(r'\d{1,6}',text))


# print(re.findall(r'\w{3,}',text))


# Wild Card


# print(re.findall(r'p...',text))

text='this is cat but not that. i want hat and cat both'


# print(re.findall(r'.a.',text))

text="Hi Thank You For The Gift 2-1 lol-lol "

# print(re.findall(r'\d$',text))


### Exclusion


# print(re.findall(r'[^\W]+',text))

# print(re.findall(r'[^\D]+',text))


# print(re.findall(r'[\w]+-[\w]+',text))



text="Google Announced a new pixel at Google I/O . Google I/O is a great place to learn and get all updates"


# pattern=[{'TEXT':'Google'},{'TEXT':'I'},{'TEXT':'/'},{'TEXT':'O'}]
pattern=[{'TEXT':'Google'},{'TEXT':'I','OP':'?'},{'TEXT':'/','OP':'?'},{'TEXT':'O','OP':'?'}]
def callBack_method(matcher,doc,i,matches):
    match_id,start,end=matches[i]
    entity=doc[start:end]
    print(entity.text)


matcher=Matcher(nlp.vocab)
matcher.add('Google',callBack_method,pattern)
doc=nlp(text)
print(matcher(doc))


#Find Word Google
# pattern=[{'TEXT':'Google'},{'TEXT':'I','OP':'?'},{'TEXT':'/','OP':'?'},{'TEXT':'O','OP':'?'}]




