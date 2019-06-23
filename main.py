import tensorflow as tf
import warnings
import os
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
from data.opp115 import OPP115
from ml.model import Model
from ml.mnb import MNB
from ml.svm import SVM
from ml.cnn import CNN
from gensim.models import KeyedVectors as kv


def load_vectors(name):
    print('\n# VECTORS\n\nloading %s vectors...' % name)

    vectors = None

    if name == 'acl1010':
        vectors = kv.load('data/acl1010/acl1010.vec', mmap='r')

    elif name == 'google':
        vectors = kv.load_word2vec_format('data/google/google-news.bin', binary=True)
    else:
        print('Error loading vectors')

    if vectors is not None:
        print('loaded %s vectors' % (len(vectors.wv.vocab)))

    return vectors


# disable warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# OPP-115
opp115 = OPP115()
data = opp115.consolidated.merge(opp115.encoded, on=['policy_id', 'segment_id']).drop_duplicates()
# opp115.display_statistics()

# Vectors
vectors = load_vectors('acl1010')

# MNB
# mnb = MNB(opp115)
# mnb.cross_validate('')
# mnb.tune_hyperparameters('kfold')

# SVM
# svm = SVM(opp115)
# svm.cross_validate()
# svm.tune_hyperparameters('kfold')

# CNN
x = data['segment']
y = np.argmax(data[Model.LABELS].values, axis=1)

cnn = CNN(vectors)
x = cnn.create_sequences(x)  # idk if this should be in a separate class
cnn.create()
cnn.evaluate(x, y)