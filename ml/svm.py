from ml.model import Model
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class SVM(Model):

    def __init__(self):
        print('\n# SVM\n')

        self.binary = True

        def create(self):
            svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=100, tol=0.001, random_state=42)
            self.model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', svm)])
