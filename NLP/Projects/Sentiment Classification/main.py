import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix 
import spacy
from  spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.svm import LinearSVC
import string


punct=string.punctuation
nlp=spacy.load('en_core_web_sm')
stopwords=list(STOP_WORDS)




data_yelp=pd.read_csv('db/yelp_labelled.txt',sep='\t',header=None)
# print(data_yelp.head())

columns_name=['Review','Sentiment']
data_yelp.columns=columns_name

# print(data_yelp.head())
# print(data_yelp.shape)

data_amazon=pd.read_csv('db/amazon_cells_labelled.txt',sep='\t',header=None)
data_amazon.columns=columns_name
# print(data_amazon.head())
# print(data_amazon.shape)

data_imdb=pd.read_csv('db/imdb_labelled.txt',sep='\t',header=None)
data_imdb.columns=columns_name
# print(data_imdb.head())
# print(data_imdb.shape)


data=data_yelp.append([data_amazon,data_imdb],ignore_index=True)

data.columns=columns_name
# print(data)

# print(data['Sentiment'].value_counts())
# print(data.isnull().sum())

# Tokenization


# print(punct)


def text_data_cleaning(sentences):
    doc=nlp(sentences)
    tokens=[]
    for token in doc:
        if(token.lemma_ != "-PRON-"):
            temp=token.lemma_.lower().strip()
        else:
            temp=token.lower_
        tokens.append(temp)
    cleaned_token=[]
    for token in tokens:
        if(token not in stopwords and token not in punct):
            cleaned_token.append(token)
    return cleaned_token

# print(text_data_cleaning("hello how are you??. I am amit"))


# Vectorization feature Engineering TF-IDF




tfidf=TfidfVectorizer(tokenizer=text_data_cleaning)

classifier=LinearSVC()

x=data['Review']
y=data['Sentiment']


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# print(X_train.shape,X_test.shape)

clf=Pipeline([('tfidf',tfidf),('clf',classifier)])

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# print(clf.predict(['Woow, This is amazing Product']))

print(clf.predict(['Such A Weaste of my time,but beautiful to watch']))