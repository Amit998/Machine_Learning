from nltk.corpus.reader import reviews
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


data=pd.read_csv('D:/study/datasets/CSV_data/Fake-News-Classifier-master/train.csv',sep=',')

x=data.drop('label',axis=1)
y=data['label']


df=data.dropna()

message=df.copy()
message.reset_index(inplace=True)


corpus=[]

for i in range(0,len(message)):
    review=re.sub('[^a-zA-Z]',' ',message['text'][i])
    review=review.lower()
    review=review.split()

    review=[ ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)




cv=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()
y=message['label']


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


count_df=pd.DataFrame(X_train,columns=cv.get_feature_names())

# print(count_df.head())

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()



from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train,Y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(Y_test,pred)
print(score)
cm=metrics.confusion_matrix(Y_test,pred)
print(cm) 
