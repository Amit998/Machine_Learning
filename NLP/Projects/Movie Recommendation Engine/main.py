import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("D:/study/datasets/CSV_data/movie_dataset.csv")
def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]
def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]
# print(df.columns)
features=['genres','keywords','cast','original_language','director']

for feature in features:
    df[feature]=df[feature].fillna('')

def combine_features(row):
    try:
        return row['genres'] + " "+row['keywords'] + " "+ row['cast'] + " "+row['director']
    except :
        print('ERROR',row)

df["combine_features"]=df.apply(combine_features,axis=1)

cv=CountVectorizer()

count_matrix=cv.fit_transform(df["combine_features"]).toarray()

# print(count_matrix[0])
cosin_sim=cosine_similarity(count_matrix)

# print(cosin_sim)
movie=input()
# movie='Maze Runner: The Scorch Trials'

movie_index=get_index_from_title(movie)
# print(movie_index)

similer_movies=list(enumerate(cosin_sim[movie_index]))
sorted_similer_movies=sorted(similer_movies,key=lambda x:x[1],reverse=True)
# print(sorted_similer_movies)


# print(df["combine_features"].head())
i=0
for movie in sorted_similer_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if (i ==10):
        break