from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

df_train = pd.read_csv('amazon_baby_train.csv.gz', compression = "gzip", keep_default_na = False, na_values = {'review': ""})
df_test  = pd.read_csv('amazon_baby_test.csv.gz',  compression = "gzip", keep_default_na = False, na_values = {'review': ""})

# skip neutral sentiments
df_train = df_train[df_train['rating'] != 3]
df_test  = df_test[df_test['rating'] != 3]

vectorizer = text.CountVectorizer(vocabulary = selected_words)



print("---------- Queston 1,2 ----------")

doc_term_matrix_train = vectorizer.fit_transform(df_train['review']).toarray()
sentiment_train = df_train['rating'] > 3

for feature, count in zip(vectorizer.get_feature_names(), np.sum(doc_term_matrix_train, axis=0)):
	print("%10s: %8d" % (feature, count))



print("---------- Queston 3,4 ----------")

model = LogisticRegression()
model.fit(doc_term_matrix_train, sentiment_train)

for feature, weight in zip(vectorizer.get_feature_names(), model.coef_[0]):
	print("%10s: %10f" % (feature, weight))



print("---------- Queston 5,6 ----------")

doc_term_matrix_test = vectorizer.fit_transform(df_test['review']).toarray()
sentiment_test = df_test['rating'] > 3

print("score: %f" % model.score(doc_term_matrix_test, sentiment_test))

print("confusion matrix:")
print(confusion_matrix(sentiment_test, model.predict(doc_term_matrix_test)))

print("classification report:")
print(classification_report(sentiment_test, model.predict(doc_term_matrix_test)))



print("---------- Queston 10 ----------")

max_item_review = df_test[df_test['name'] == 'Baby Trend Diaper Champ'].max()
x = vectorizer.fit_transform([max_item_review['review']]).toarray()

print(model.predict_proba(x))