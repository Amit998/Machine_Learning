import pandas as pd

messages=pd.read_csv('data.tsv',sep='\t',names=['label','message'])

# print(message.columns)

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]',' ',str(messages['message'][i]))
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# print(len(corpus))



# Creating the Bag of Words model
# from sklearn.feature_extraction.text import TfidfVectorizer
# cv = TfidfVectorizer(max_features=2500)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)

X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# print('done')


# Train Test Split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

# print(X_test.shape)

# print(y_pred)
true=0
false=0
for i in range(len(y_pred)):
    if (y_pred[i]==y_test[i]):
        true+=1
    else:
        false+=1
# print(true , ' True ' , false ,' false')





def predict_this(val):
    review = re.sub('[^a-zA-Z]',' ',str(val))
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    
    # print(review)

    cv = CountVectorizer(max_features=2500)

    # X = cv.fit_transform([review]).toarray()
    
    
    print(X.shape)



    # detector=spam_detect_model.predict(X)

    # return detector
# predict_this('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')
print(predict_this(['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...']))

