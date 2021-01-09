from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import  cosine_similarity


text=["London Paris London","Paris London Paris"]


cv=CountVectorizer()
x=cv.fit_transform(text).toarray()

print(x)

print(cosine_similarity(x))
