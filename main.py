def warn(*args, **kwargs):
    pass

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.warn = warn
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import opp115
import net
import classical
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from time import time

data = opp115.load()
data = data[data['data_practice'] != 'Other']

x = data['text'].values
y = pd.get_dummies(data['data_practice']).values # net
#y = data['data_practice'].values # classical

params = {
    'memory_dim': [100],
    'embedding_dropout': [0.0, 0.5],
    'lstm_layers': [1, 2],
    'lstm_dropout': [0.0, 0.1, 0.25, 0.5],
    'recurrent_dropout': [0.0, 0.1, 0.25, 0.5],
    'optimizer': ['adam', 'adam_0.01', 'nadam', 'rmsprop']
}

net.tune(x, y, 'lstm', params)

# tracing back of cnn
'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
vocab = tokenizer.word_index

x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=100)
    
vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
matrix = net.init_matrix(vocab, vec)

cnn = net.cnn(matrix, len(vocab) + 1, {})
# TODO: cnn.fit()

dense_outputs = net.get_layer_output(4, cnn, x)
pooling_outputs = net.get_layer_output(2, cnn, x)
conv_outputs = net.get_layer_output(1, cnn, x)
embedding_outputs = net.get_layer_output(0, cnn, x)

for i in range(dense_outputs.shape[0]):
    dense_output = dense_outputs[i]
    data_practice = np.argmax(dense_output) # index of largest value in dense layer (predicted data practice)
    print(data_practice)

    pooling_output = pooling_outputs[i]
    pooling_largest = np.argmax(pooling_output)

    conv_output = conv_outputs[i]

    embedding_output = embedding_outputs[i][pooling_largest]
    print(embedding_output)

    print(vec.most_similar(positive=[embedding_output]))
    break
'''

# classical tuning
'''
params = {
    'penalty': ['l2'],
    'tol': [1],
    'c': [1.5],
}

classical.tune(x, y, 'lr', params)
'''

# lstm tuning
'''
params = {
    'memory_dim': [100],
    'embedding_dropout': [0.0, 0.5],
    'lstm_layers': [1, 2, 3],
    'lstm_dropout': [0.0, 0.1, 0.25, 0.5],
    'recurrent_dropout': [0.0, 0.1, 0.25, 0.5],
    'optimizer': ['adam', 'adam_0.01', 'nadam', 'rmsprop']
}

net.tune(x, y, 'lstm', params)
'''

# bi_lstm tuning
'''
params = {
    'memory_dim': [100],
    'embedding_dropout': [0.0, 0.5],
    'lstm_layers': [1, 2, 3],
    'lstm_dropout': [0.0, 0.1, 0.25, 0.5],
    'recurrent_dropout': [0.0, 0.1, 0.25, 0.5],
    'optimizer': ['adam', 'adam_0.01', 'nadam', 'rmsprop']
}

net.tune(x, y, 'bi_lstm', params)
'''

# cnn_lstm tuning
'''
params = {
    'memory_dim': [100],
    'embedding_dropout': [0.0, 0.5],
    'lstm_layers': [1, 2, 3],
    'lstm_dropout': [0.0, 0.1, 0.25, 0.5],
    'recurrent_dropout': [0.0, 0.1, 0.25, 0.5],
    'optimizer': ['adam', 'adam_0.01', 'nadam', 'rmsprop']
}

net.tune(x, y, 'cnn_lstm', params)
'''