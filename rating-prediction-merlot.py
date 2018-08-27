def warn(*args, **kwargs):
    pass


import warnings
from random import random

import pandas as pd
import plotly
import plotly.graph_objs as go
from sklearn import tree
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from stop_words import get_stop_words

warnings.warn = warn

df = pd.read_csv('comments_raw.csv')

classifiers = {
    "Naïve Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}
max_features = [200, 1000, 10000]
n_categories = {
    "Exc./Non-Exc": "nota",
    "1-5": "classe 2"
}

n_grams = {
    "1": (1, 1),
    "1-3": (1, 3)
}
stop_words = {
    "No": None,
    "Yes": ENGLISH_STOP_WORDS.union(get_stop_words('spanish')).union(get_stop_words('portuguese'))
}

graph_data = []
for category_key, category in n_categories.items():
    for max_feature in max_features:
        for gram_key, gram in n_grams.items():
            for stop_key, stop_word in stop_words.items():
                for class_key, classifier in classifiers.items():
                    count_vectorizer = CountVectorizer(
                        analyzer='word',
                        lowercase=True,
                        stop_words=stop_word,
                        ngram_range=gram,
                        max_features=max_feature
                    )

                    features = count_vectorizer.fit_transform(
                        df["comentario"]
                    )

                    features_nd = features.toarray()

                    X_train, X_test, y_train, y_test = train_test_split(
                        features_nd,
                        df[category],
                        train_size=0.80
                    )

                    predictor = classifier.fit(X=X_train, y=y_train)
                    y_pred = predictor.predict(X_test)

                    new_line = [class_key, "Count", stop_key, gram_key,
                                max_feature, category_key, accuracy_score(y_test, y_pred)*100]
                    print(new_line)
                    graph_data.append(new_line)

                    tfidf_vectorizer = TfidfVectorizer(
                        sublinear_tf=True,
                        max_df=0.5,
                        stop_words=stop_word,
                        ngram_range=gram,
                        max_features=max_feature
                    )

                    features = tfidf_vectorizer.fit_transform(
                        df["comentario"]
                    )

                    features_nd = features.toarray()

                    X_train, X_test, y_train, y_test = train_test_split(
                        features_nd,
                        df[category],
                        train_size=0.80
                    )

                    predictor = classifier.fit(X=X_train, y=y_train)
                    y_pred = predictor.predict(X_test)

                    new_line = [class_key, "TF-IDF", stop_key, gram_key,
                                max_feature, category_key, accuracy_score(y_test, y_pred)*100]
                    print(new_line)
                    graph_data.append(new_line)

df_data = pd.DataFrame(graph_data, columns=["Classifier",
                                            "Word Vectorization",
                                            "Stop-Words",
                                            "N-Grams",
                                            "Max Features",
                                            "Categories",
                                            "Accuracy (%)"])

trace = go.Table(
    header=dict(values=list(df_data.columns),
                fill=dict(color='#C2D4FF'),
                align=['left'] * 5),
    cells=dict(values=[df_data["Classifier"],
                       df_data["Word Vectorization"],
                       df_data["Stop-Words"],
                       df_data["N-Grams"],
                       df_data["Max Features"],
                       df_data["Categories"],
                       df_data["Accuracy (%)"]],
               fill=dict(color='#F5F8FF'),
               align=['left'] * 5))

data = [trace]
plotly.offline.plot(data, filename='pandas_table')
