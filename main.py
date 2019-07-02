import os
import tensorflow as tf
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from data.corpus import Corpus
from ml.cnn import CNN

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

corpus = Corpus()

opp115 = corpus.load()

#statistics = corpus.generate_statistics()
# print(statistics)

cnn = CNN(opp115)
# cnn.evaluate()
cnn.tune()
