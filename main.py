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

net.evaluate(x, y, 'lstm', {'memory_dim': 100})

'''tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
vocab = tokenizer.word_index
print('found %s unique tokens' % len(tokenizer.word_index))

x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=100)
    
vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
matrix = net.init_matrix(vocab, vec)

lstm = net.lstm(matrix, len(vocab) + 1, {'memory_dim': 100})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

lstm.fit(x, y, batch_size=64, epochs=50)

y_predict = lstm.predict(x_test)

y_predict, y_test = np.argmax(y_predict, axis=1), np.argmax(y_test, axis=1)

print(classification_report(y_test, y_predict))'''

#cnn = net.cnn(matrix, len(vocab) + 1)

'''e = net.get_layer_output(4, cnn, x)[0]
print(e) # final layer output
index = np.argmax(e)
l = e[index]
print(l)'''

'''d = net.get_layer_output(3, cnn, x)[0]
print(d)
print(np.array_split(d, 10)[index])'''

#c = net.get_layer_output(2, cnn, x)[0]
#print(c)
#print(net.get_layer_output(2, cnn, x)[0])

'''b = net.get_layer_output(1, cnn, x)[0]
print(b)
print(b.shape)'''

#a = net.get_layer_output(0, cnn, x)[0]
#print(a[10])
#print(a[0].shape)
#print(net.get_layer_output(1, cnn, x)[0])


'''print(x)
print(x.shape)
x = tokenizer.sequences_to_texts(x)'''


'''
params = {
    'penalty': ['l2'],
    'tol': [1],
    'c': [1.5],
}

classical.tune(x, y, 'lr', params)
'''