from sklearn.feature_extraction.text import CountVectorizer

text=["You Are One Of the greatest Person in the world"]

vectorizer=CountVectorizer()

vectorizer=vectorizer.fit(text)

# print(vectorizer.vocabulary_)

vector=vectorizer.transform(text)

# print(vector.shape)
# print(type(vector))
# print(vector.toarray())



import  numpy as np
import re

def token_sententes(sentence):
    words=[]
    for sentence in sentence:
        w = extract_words(sentence)
        words.extend(w)
    words=sorted(list(set(words)))
    return words


def extract_words(sentence):
    ignore_words=['a']
    words=re.sub("[^w]"," ", sentence).split()
    words_cleaned=[w.lower()  for w in words if w not in ignore_words]
    return words_cleaned


def bagOfWord(sentence,words):
    sentence_word=extract_words(sentence)
    print(words)
    bag=np.zeros(len(words))
    for sw in sentence_word:
        for i,word in enumerate(words):
            if(word == sw):
                bag[i] +=1

    return np.array(bag)


text=["Machine Learning is Great","You Are One Of the greatest Person in the world"]
vocabulary=token_sententes(text)

# print(vocabulary)
bagOfWord("Machine Learning is Great",vocabulary)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None)

train_data_features=vectorizer.fit_transform(text)

vectorizer.transform(["Machine Learning is Great","You Are One Of the greatest Person in the world"]).toarray()
