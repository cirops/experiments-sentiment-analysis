import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report


### Pre-process Data ###
df = pd.read_csv('data/proc/comments_lang.csv')
df = df.drop(columns=['classe 1', 'nota'])
df.columns=['comment', 'rating']
df['comment'] = df['comment'].str.lower()
df['comment'] = df['comment'].str.replace(r'http\S+\b', '')
df["rating"] = df["rating"].replace({'nao excelente': 0, 'excelente': 1})


### Grid Search Parameters ###
max_features = {
    "None": None,
    "200": 200,
    "500": 500,
    "1000": 1000,
    "5000": 5000,
    "10000": 10000
}

n_grams = {
    "1": (1, 1),
    "2": (2, 2),
    "3": (3, 3),
    "1-2": (1, 2),
    "1-3": (1, 3),
    "2-3": (2, 3)
}

stop_words = {
    "No": None,
    "Yes": ENGLISH_STOP_WORDS
}

n_categories = {
    "Exc./Non-Exc": "rating_b",
    "1-5": "rating"
}

vectorizers = {
    "Count": CountVectorizer,
    "TF-IDF": TfidfVectorizer
}

classifiers = {
    "Naïve Bayes": MultinomialNB(),
    "Support Vector Machine": CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=4000)
}

n_tests = len(max_features)*len(n_grams)*len(stop_words)*len(vectorizers)*len(classifiers)
print("Number of tests: ", n_tests)

docs_train, docs_test, y_train, y_test = train_test_split(
    df['comment'], df['rating'], test_size=0.25, random_state=None)

data = []
count = 1

### Grid Search ###
for feat_key, max_feature in max_features.items():
    for gram_key, gram in n_grams.items():
        for stop_key, stop_word in stop_words.items():
            for class_key, clf in classifiers.items():
                for vec_key, vec in vectorizers.items():
                    time_start = time.time()

                    vectorizer = vec(
                        analyzer='word',
                        lowercase=True,
                        stop_words=stop_word,
                        ngram_range=gram,
                        max_features=max_feature
                    )

                    X_train = vectorizer.fit_transform(docs_train)
                    X_test = vectorizer.transform(docs_test)
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)
                    y_pred_prob = clf.predict_proba(X_test)[:, 1]
                    accuracy = accuracy_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_prob)
                    report = classification_report(y_test, y_pred,  output_dict=True)
                    kappa = cohen_kappa_score(y_test, y_pred)
                    line = [
                        '{0:.2f}'.format(time.time() - time_start),
                        class_key,
                        feat_key,
                        gram_key,
                        stop_key,
                        vec_key,
                        '{0:.2f}'.format(accuracy),
                        '{0:.2f}'.format(report['0']['precision']),
                        '{0:.2f}'.format(report['1']['precision']),
                        '{0:.2f}'.format(report['0']['recall']),
                        '{0:.2f}'.format(report['1']['recall']),
                        '{0:.2f}'.format( report['0']['f1-score']),
                        '{0:.2f}'.format(report['1']['f1-score']),
                        '{0:.2f}'.format(kappa),
                        '{0:.2f}'.format(roc_auc)
                    ]
                    print(count, "/", n_tests, ": ", line)
                    count += 1
                    data.append(line)

### Dumping Results to csv ###
df = pd.DataFrame(data, columns=[
    'Time Elapsed',
    'Classifier',
    'Number of Features',
    'N-gram Range',
    'Stop-words',
    'Vectorizer',
    'Accuracy (%)',
    'Recall (Negative)',
    'Recall (Positive)',
    'Precision (Negative)',
    'Precision (Positive)',
    'F1 (Negative)',
    'F1 (Positive)',
    'Kappa',
    'ROC AUC'])

df.to_csv('results.csv', index=False)