from ml.model import Model
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class SVM(Model):

    def __init__(self):
        print('\n# SVM\n')

        self.binary = True
        self.alpha = 0.0001
        self.iterations = 100
        self.tolerance = 0.001

    def create(self):
        svm = SGDClassifier(loss='hinge', penalty='l2', alpha=self.alpha, max_iter=self.iterations, tol=self.tolerance, random_state=42)
        self.model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', svm)])
