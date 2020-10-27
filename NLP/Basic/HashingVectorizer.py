from sklearn.feature_extraction.text import HashingVectorizer

doc2=["Apple to Build hong kong factory cost $5 milion "]


vectorizer=HashingVectorizer(n_features=20)

vector=vectorizer.transform(doc2)



print(vector.shape)
print(vector.toarray())