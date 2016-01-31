import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import pairwise_distances


def top_words(values, features, top = 1):
    return features[values.toarray()[0].argsort()[-top:]]



df = pd.read_csv('people_wiki.csv')



print("================= Data preparation =================")

print("computing word count")
count_vectorizer = CountVectorizer()

word_count_matrix = count_vectorizer.fit_transform(df['text'])
word_count_features = np.array(count_vectorizer.get_feature_names())



print("computing TF-IDF")
tfidf_vectorizer = TfidfTransformer()

word_tfidf_matrix = tfidf_vectorizer.fit_transform(word_count_matrix)
word_tfidf_features = word_count_features



print("===================== Queston 1 =====================")


idx = df[df['name'] == 'Elton John'].index.tolist()[0]
top = top_words(word_count_matrix[idx], word_count_features, 3)

print(", ".join(top))



print("===================== Queston 2 =====================")

idx = df[df['name'] == 'Elton John'].index.tolist()[0]
top = top_words(word_tfidf_matrix[idx], word_tfidf_features, 3)

print(", ".join(top))



print("===================== Queston 3 =====================")

idx_1 = df[df['name'] == 'Elton John'].index.tolist()[0]
idx_2 = df[df['name'] == 'Victoria Beckham'].index.tolist()[0]

dist = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix[idx_2], metric = 'cosine')

print("cosine distance: %.2f" % dist)



print("===================== Queston 4 =====================")

idx_1 = df[df['name'] == 'Elton John'].index.tolist()[0]
idx_2 = df[df['name'] == 'Paul McCartney'].index.tolist()[0]

dist = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix[idx_2], metric = 'cosine')

print("cosine distance: %.2f" % dist)



print("===================== Queston 5 =====================")

idx_1 = df[df['name'] == 'Elton John'].index.tolist()[0]
idx_2 = df[df['name'] == 'Victoria Beckham'].index.tolist()[0]
idx_3 = df[df['name'] == 'Paul McCartney'].index.tolist()[0]

dist_1 = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix[idx_2], metric = 'cosine')
dist_2 = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix[idx_3], metric = 'cosine')

print("cosine distance 1: %.2f" % dist_1)
print("cosine distance 2: %.2f" % dist_2)



print("===================== Queston 6 =====================")

idx_1 = df[df['name'] == 'Elton John'].index.tolist()[0]
idx_min = pairwise_distances(word_count_matrix[idx_1], word_count_matrix, metric = 'cosine')[0].argsort()[1]

print("closest: %s" % df.iloc[idx_min]['name'])



print("===================== Queston 7 =====================")

idx_1 = df[df['name'] == 'Elton John'].index.tolist()[0]
idx_min = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix, metric = 'cosine')[0].argsort()[1]

print("closest: %s" % df.iloc[idx_min]['name'])



print("===================== Queston 8 =====================")

idx_1 = df[df['name'] == 'Victoria Beckham'].index.tolist()[0]
idx_min = pairwise_distances(word_count_matrix[idx_1], word_count_matrix, metric = 'cosine')[0].argsort()[1]

print("closest: %s" % df.iloc[idx_min]['name'])



print("===================== Queston 9 =====================")

idx_1 = df[df['name'] == 'Victoria Beckham'].index.tolist()[0]
idx_min = pairwise_distances(word_tfidf_matrix[idx_1], word_tfidf_matrix, metric = 'cosine')[0].argsort()[1]

print("closest: %s" % df.iloc[idx_min]['name'])