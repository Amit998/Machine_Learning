from nltk.corpus.reader import reviews
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


data=pd.read_csv('D:/study/datasets/CSV_data/Fake-News-Classifier-master/train.csv',sep=',')



# print(data.head())




# print(x.shape,y.shape)

df=data.dropna()
# print(df.shape)

message=df.copy()
message.reset_index(inplace=True)

# print(message.head(10))




corpus=[]

for i in range(0,len(message)):
    review=re.sub('[^a-zA-Z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()

    review=[ ps.stem(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
# print(corpus)


cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()
y=message['label']

# print(x.shape)
# print(y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# print(cv.get_params())

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
# print(score)
cm=metrics.confusion_matrix(Y_test,pred)
# print(cm) 


from sklearn.linear_model import PassiveAggressiveClassifier
linea_clf=PassiveAggressiveClassifier(n_jobs=50)

linea_clf.fit(X_train,Y_train)
pred=linea_clf.predict(X_test)

score=metrics.accuracy_score(Y_test,pred)
# print(score)
cm=metrics.confusion_matrix(Y_test,pred)
# print(cm)

classifier=MultinomialNB(alpha=0.1)
classifier.fit(X_test,Y_test)
previous_score=0

# for alpha in np.arange(0,1,0.1):
#     print(alpha)
    # sub_classifier=MultinomialNB(alpha=alpha)
    # sub_classifier.fit(X_train,Y_train)
    # pred=linea_clf.predict(X_test)

    # score=metrics.accuracy_score(X_test,pred)

    # if (score > previous_score):
    #     classifier=sub_classifier
    # print("Alpha : {} , Score : {} ".format(alpha,score))

    


feature_names=cv.get_feature_names()

# print(classifier.coef_[0])
# print(sorted(zip(classifier.coef_[0],feature_names),reverse=True)[:20])
# print(sorted(zip(classifier.coef_[0],feature_names),reverse=True)[-20:])