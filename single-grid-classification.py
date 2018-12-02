import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/proc/comments_lang.csv')
df = df.drop(columns=['classe 1', 'nota'])
df.columns=['comment', 'rating']
df['comment'] = df['comment'].str.lower()
df['comment'] = df['comment'].str.replace(r'http\S+\b', '')
df["rating"] = df["rating"].replace({'nao excelente': 0, 'excelente': 1})

docs_train, docs_test, y_train, y_test = train_test_split(
    df['comment'], df['rating'], test_size=0.25, random_state=None)

parameters = {  
    'vec__max_df': (0.5, 0.75, 1.0),  
    'vec__min_df': (1, 10, 50),  
    'tfidf__use_idf': (True, False),  
    'tfidf__sublinear_tf': (True, False),
    'tfidf__norm': ('l1', 'l2'),  
    'clf__C': (10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
    # 'clf__dual': (True, False),
    # 'clf__fit_intercept': (True, False),
    # 'clf__loss': ('hinge', 'squared_hinge'),
    'clf__tol': (1e-5, 1e-4, 1e-3),
}

pipeline = Pipeline([
    ('vec',   CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf',   LinearSVC(max_iter=10000))
])

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=51, scoring='roc_auc')

if __name__ == "__main__":
    grid_search.fit(docs_train, y_train)
    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (grid_search.cv_results_['params'][i],
                    grid_search.cv_results_['mean_test_score'][i],
                    grid_search.cv_results_['std_test_score'][i]))

    y_predicted = grid_search.predict(docs_test)

    print(classification_report(y_test, y_predicted))

    cm = confusion_matrix(y_test, y_predicted)
    print(cm)


    print(grid_search.best_score_)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, grid_search.best_params_[param_name]))

