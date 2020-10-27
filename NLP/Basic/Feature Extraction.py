
# IDF=Inverse Document Frequency
# TF=Term Frequency

# TF=Term i Frequency in Document/total words in document

# IDF=log2(Total Document/documents with term i)

# t=term
# j=document

from sklearn.feature_extraction.text import CountVectorizer

text=["In case you had successfully completed the test without any issues"," you can ignore this mail. Your previous score will be considered"]
vectorizer=CountVectorizer()

vectorizer.fit(text)

print(vectorizer.vocabulary_)

newVector=vectorizer.transform(text)

print(newVector.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer()
vectorizer.fit(text)

print(vectorizer.idf_)
print(vectorizer.vocabulary_)
