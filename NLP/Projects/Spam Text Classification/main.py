# doc1="I Am High"
# doc2="Yes I Am High"
# doc3="I am Kidding"


# Bag Of Words Count the occarance Of A Word


import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import random

df=pd.read_csv("spam.csv", encoding = "latin-1")

# print(df.sample(10))

# print(len(df))

# print(df['label'].value_counts())

# Balance this Data

doc1="I Am High"

# doc1=doc1.split()
# print(len(doc1))

df['length']=df['message'].apply( lambda x: len(x.split()) )

# print(df.head(10))



ham=df[df["label"] == 'ham']
spam=df[df["label"] == 'spam']

# print(ham.head())
# print(spam.head())




# print(data.tail())

# print(ham.shape,spam.shape)


ham=ham.sample(spam.shape[0])

data=ham.append(spam,ignore_index=True)

# print(ham.shape,spam.shape)


# Exploratort Data Analysis

# print(data[data['label']=='ham'].count())


## Exploratry Data Analysis
# plt.hist(data[data['label']=='ham']['length'],bins=100,alpha=0.7)
# plt.hist(data[data['label']=='spam']['length'],bins=100,alpha=0.7)
# plt.show()


#Below Will Not Work
# plt.hist(data[data['label']=='ham']['punct'],bins=100,alpha=0.7)
# plt.hist(data[data['label']=='spam']['punct'],bins=100,alpha=0.7)
# plt.show()



#Data Preparation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline


from sklearn.feature_extraction.text import TfidfVectorizer


X_train,x_test,y_train,y_test=train_test_split(data['message'],data['label'],test_size=0.3,random_state=0,shuffle=True,stratify=data['label'])

# print(y_train)


##Bag Of Words Creation

vectorizer=TfidfVectorizer()

# X_train=vectorizer.fit_transform(X_train)

# print(X_train.shape)


####PIPELINE AND RF 


# clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier(n_estimators=100,n_jobs=-1))])

clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',SVC(C=1000,gamma='auto'))])


clf.fit(X_train,y_train)

y_pred=clf.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))



print(clf.predict(['click this link https:\\www.lol.com']))