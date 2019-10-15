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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from time import time

def test_classical(data):
    x = data['text'].values
    y = data['data_practice'].values 

    x = classical.vectorize_data(x)

    svm = classical.svm()
    mnb = classical.mnb()
    lr = classical.lr()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    classical.train(lr, x_train, y_train)

    y_pred = lr.predict(x_test)
    print(classification_report(y_test, y_pred))

def test_net(data):
    x = data['text'].values
    y = pd.get_dummies(data['data_practice']).values # net

    x, vocab = net.init_vocab(x)

    vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    matrix = net.init_matrix(vocab, vec)

    cnn = net.cnn(matrix, len(vocab) + 1)
    lstm = net.lstm(matrix, len(vocab) + 1)

    #print(net.get_layer_output(4, cnn, x)[0]) # final layer output
    #print(net.get_layer_output(3, cnn, x)[0]) # max pooling output
    #print(net.get_layer_output(2, cnn, x)[0])
    #print(net.get_layer_output(1, cnn, x)[0])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    net.train(lstm, x_train, y_train)

    y_pred = lstm.predict(x_test)
    y_test, y_pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred))

# TODO change name to evaluate, move to net.py
def skfold(x, y, model_name, epochs):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    if model_name == 'lstm':
        x, vocab = net.init_sequences(x, padding='post')
    elif model_name == 'cnn':
        x, vocab = net.init_sequences(x, padding='post')
    
    vocab_size = len(vocab) + 1
    vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    matrix = net.init_matrix(vocab, vec)

    fold_num = 1
    actual = []
    predictions = []
    
    for train, test in skf.split(x, np.argmax(y, axis=1)):
        print('fold %s' % fold_num, end='', flush=True)
        fold_num += 1
        fold_start = time()

        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        if model_name == 'lstm':
            model = net.lstm(matrix, vocab_size)
        elif model_name =='cnn':
            model = net.cnn(matrix, vocab_size)

        model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0)

        y_predict = model.predict(x_test)
        y_test, y_predict = np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)

        actual.extend(y_test)
        predictions.extend(y_predict)

        fold_end = time()
        print('\t\t%ss' % round(fold_end - fold_start, 2))

    print(classification_report(actual, predictions))

data = opp115.load()

x = data['text'].values
y = pd.get_dummies(data['data_practice']).values # net

skfold(x, y, model_name='cnn', epochs=20)

#data = data[data['data_practice'] != 'Other']
#opp115.generate_attribute_distribution(data)

#test_classical(data)
#test_net(data)