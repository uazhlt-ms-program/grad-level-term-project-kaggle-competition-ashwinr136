import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

train_data = pd.read_csv("train.csv")
train_data = train_data.dropna(subset=['TEXT'])

X_train = train_data['TEXT']
y_train = train_data['LABEL']

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

model = LogisticRegression(solver="sag",max_iter=10000,C=13.0)
model.fit(X_train_vectorized, y_train)

np.save("movie_review_model_weights_intercept.npy", model.intercept_)
np.save("movie_review_model_weights_coef.npy", model.coef_)
np.save("count_vectorizer_vocabulary.npy", vectorizer.vocabulary_)

test_data = pd.read_csv("test.csv")
test_data = test_data.dropna(subset=['TEXT'])

model.intercept_ = np.load("movie_review_model_weights_intercept.npy")
model.coef_ = np.load("movie_review_model_weights_coef.npy")
vectorizer.vocabulary_ = np.load("count_vectorizer_vocabulary.npy", allow_pickle=True).item()

test_data['TEXT'].fillna('', inplace=True)
X_test = vectorizer.transform(test_data['TEXT'])

predicted_labels = model.predict(X_test)
test_data['LABEL'] = predicted_labels

submission = test_data[['ID', 'LABEL']]
submission.to_csv("submission.csv", index=False)
