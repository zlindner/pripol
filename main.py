import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def test_classical(data):
    x = data['text'].values
    y = data['data_practice'].values 

    x = classical.vectorize_data(x)

    svm = classical.svm()
    mnb = classical.mnb()
    lr = classical.lr()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    classical.train(mnb, x_train, y_train)

    y_pred = mnb.predict(x_test)
    print(classification_report(y_test, y_pred))

def test_net(data):
    x = data['text'].values
    y = pd.get_dummies(data['data_practice']).values # net

    x, y, vocab = net.init_vocab(x, y)

    vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    matrix = net.init_matrix(vocab, vec)

    cnn = net.cnn(matrix, vocab)
    lstm = net.lstm(matrix, vocab)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    net.train(cnn, x_train, y_train)

    y_pred = cnn.predict(x_test)
    y_test, y_pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))

# initialize data
data = opp115.load()
data = data[data['data_practice'] != 'Other']

test_classical(data)
#test_net(data)