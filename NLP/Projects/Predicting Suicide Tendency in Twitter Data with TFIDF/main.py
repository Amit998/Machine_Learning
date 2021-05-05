import pandas as pd
import numpy as np
import preprocess_kgptalkie as ps
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle


df=pd.read_csv('twitter-suicidal-intention-dataset/twitter-suicidal_data.csv')

# print(df.head())

# df['intention'].value_counts()



def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


# df['tweet']=df['tweet'].apply(lambda x: get_clean(x))


tfidf=TfidfVectorizer(max_features=20000,ngram_range=(1,3),analyzer='char')

x=tfidf.fit_transform(df['tweet'])
y=df['intention']

# print(x.shape)
# print(y.shape)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


clf=LinearSVC()

clf.fit(X_train, y_train)

# file_name="finalized_model.pkl"

# pickle.dump(clf,open(file_name,'wb'))


# clf=pickle.load(open(file_name,'rb'))

# y_pred=clf.predict(X_test)

# print(classification_report(y_test,y_pred))


x='No one cares abou me.i will die alone'

x=get_clean(x)
vec=tfidf.transform([x])

# print(vec)

print(clf.predict(vec))


x='I am soo happy'

x=get_clean(x)
vec=tfidf.transform([x])

# print(vec)

print(clf.predict(vec))