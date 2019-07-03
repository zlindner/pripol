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

# 0.83
# select top 1000 features
#descending = sorted(corpus.iindex.index, key=lambda x: [node['freq'] for node in corpus.iindex.index[x] if node['id'] == -1], reverse=True)
#features = descending[:1000]
#opp115['segment'] = opp115['segment'].apply(lambda x: ' '.join([word for word in x.split() if word in features]))


#statistics = corpus.generate_statistics()
# print(statistics)

cnn = CNN(opp115)
cnn.evaluate()
# cnn.tune()
