import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")
#print(df.columns)


features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df['combined_features'] = df.apply(combine_features, axis=1)

#print(df['combined_features'].head())

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
print(count_matrix)

