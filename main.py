import os
import tensorflow as tf
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from data.corpus import Corpus
from ml.cnn import CNN
from ml.svm import SVM
from ml.mnb import MNB

import data.vectors as vectors

# vectors.demo()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

corpus = Corpus()
opp115 = corpus.load()

# select top 1000 features
#descending = sorted(corpus.iindex.index, key=lambda x: [node['freq'] for node in corpus.iindex.index[x] if node['id'] == -1], reverse=True)
#features = descending[:1000]
#opp115['segment'] = opp115['segment'].apply(lambda x: ' '.join([word for word in x.split() if word in features]))

#statistics = corpus.generate_statistics()
# print(statistics)

cnn = CNN(opp115)
cnn.evaluate(num_filters=200, ngrams=[2, 3], save_results=False)

#svm = SVM(opp115)
#svm.evaluate(alpha=0.0001, iterations=100, tolerance=0.001, save_results=False)

#mnb = MNB(opp115)
#mnb.evaluate(alpha=0.1, save_results=False)
