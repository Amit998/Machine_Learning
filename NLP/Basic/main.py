import re
import string
# text = 'I\'m am with you for the entire life in U.K'
# words = re.split(r'\W+',text)
# print(words[:100])

# words=text.split()
# re_punc=re.compile('[%s]'%re.escape(string.punctuation))
# stripped=[re_punc.sub('',w) for w in words]
# print(stripped[:100])

# re_print=re.compile('[^%s]'%re.escape(string.printable))
# result=[re_print.sub('',w) for w in words]
# words=[word.lower() for word in words]
# print(words)

import spacy
nlp=spacy.load('en_core_web_sm')

string= 'I\'m am with you for the entire life in U.K'
string ="My Income is 100$."

doc=nlp(string)
# for token in doc:
#     print(token.text, end='|')
# for t in doc:
#     print(t.text)
# for t in doc:
#     print(t)

print(len(doc))
# print(doc.vocab)
# print(doc[-2:])

doc2=nlp("Apple to Build hong kong factory cost $5 milion ")

# for ent in doc2.ents:
#     print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))


# doc3=nlp("The dataset we will use comes from a Pubmed search, and contains 1748 observations and 3 variables, as described below:")
# for chunk in doc3.noun_chunks:
#     print(chunk.text)

from spacy import displacy
string= nlp('I\'m am with you for the entire life in U.K')
# displacy.render(string,style='dep')
displacy.serve(string,style='ent')