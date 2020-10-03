import nltk
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



# print(os.listdir(nltk.data.find('corpora')))


from nltk.corpus import movie_reviews

neg_review=movie_reviews.fileids('neg')
pos_review=movie_reviews.fileids('pos')

# print(len(neg_review))
# print(len(pos_review))
# print(movie_reviews.fileids('pos'))
# print(' ')
# print(movie_reviews.fileids('pos'))

rev= nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
# print(rev)
rev_list=[]
for rev in neg_review:
    rev_text_neg=nltk.corpus.movie_reviews.words(rev)
    review_one_string=" ".join(rev_text_neg)
    review_one_string =review_one_string.replace(' ,',',')
    review_one_string =review_one_string.replace(' .','.')
    review_one_string =review_one_string.replace("\' ","'")
    review_one_string =review_one_string.replace(" \'","'")
    rev_list.append(review_one_string)
print(len(rev_list))
# pos_review=movie_reviews.fileids('pos')
neg_targets=np.zeros((1000,),dtype=np.int)
pos_targets=np.ones((1000,),dtype=np.int)

target_list=[]
for neg_target in neg_targets:
    target_list.append(neg_target)
for pos_target in pos_targets:
    target_list.append(pos_target)
print(len(target_list))

y=pd.Series(target_list)
print(type(y))

print(y.head())

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer(lowercase=True,stop_words='english',min_df=2)

x_count_vect=count_vect.fit_transform(rev_list)
print(x_count_vect.shape)

x_names=count_vect.get_feature_names()
# print(x_names)


x_count_vect=pd.DataFrame(x_count_vect.toarray(),columns=x_names)
print(x_count_vect.shape)


from sklearn import metrics
from sklearn.metrics import confusion_matrix


import numpy as np
from sklearn.model_selection import train_test_split

print(x_count_vect.shape)
print(y.shape)


x_train_cv,x_test_cv,y_train_cv,y_test_cv=train_test_split(x_count_vect,y,test_size=0.25,random_state=5)


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred_gnb=gnb.fit(x_train_cv,y_train_cv).predict(x_test_cv)

from sklearn.naive_bayes import MultinomialNB
clf_cv=MultinomialNB()
clf_cv.fit(x_train_cv,y_train_cv)


y_pred_cv=clf_cv.predict(x_test_cv)


print(metrics.accuracy_score(y_test_cv,y_pred_cv))

print(confusion_matrix(y_test_cv,y_pred_cv))