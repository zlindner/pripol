from ml.model import Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


class MNB(Model):

    def __init__(self):
        print('\n# MNB\n')

        self.binary = True
        self.alpha = 0.1

    def create(self):
        mnb = MultinomialNB(alpha=0.1)
        self.model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', mnb)])
