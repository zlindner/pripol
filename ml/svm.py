import numpy as np
import time

from data.corpus import Corpus
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


class SVM():

    def __init__(self, opp115):
        self.corpus = opp115

    def create(self, alpha, iterations, tolerance):
        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha, max_iter=iterations, tol=tolerance, random_state=42)
        model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', svm)])

        return model

    def evaluate(self, alpha, iterations, tolerance, save_results=False):
        x = self.corpus['segment']
        y = self.corpus[Corpus.DATA_PRACTICES].values

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        fold = 1  # current fold
        true = []
        pred = []
        times = []  # list to elapsed time for each fold

        y = np.argmax(y, axis=1)

        for train, test in skf.split(x, y):
            print('Fold #%s...' % fold, end='', flush=True)
            fold += 1
            fold_start = time.time()

            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]

            model = self.create(alpha, iterations, tolerance)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            true.extend(y_test)
            pred.extend(y_pred)

            fold_end = time.time()
            elapsed = round(fold_end - fold_start, 2)
            times.append(elapsed)
            print('%ss' % elapsed)  # elapsed time for each fold

        results = classification_report(true, pred, target_names=Corpus.DATA_PRACTICES)  # accuracy = micro avg

        print(results)
        print('Total time elapsed: %ss\n' % np.round(sum(times), 2))

        if save_results:
            with open('results/svm_results.txt', 'a') as f:
                f.write('alpha: %s, iterations: %s, tolerance: %s\n\n%s\nTotal time elapsed: %ss\n\n' % (alpha, iterations, tolerance, results, sum(times)))
