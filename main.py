import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import opp115
import pandas as pd
import numpy as np
from nets import cnn, lstm
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = opp115.load()
data = data[data['data_practice'] != 'Other']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].values)
vocab = tokenizer.word_index
vocab_size = len(vocab) + 1
print('found %s unique tokens' % len(tokenizer.word_index))

x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x, maxlen=200)
print('data tensor shape: ', x.shape)

y = pd.get_dummies(data['data_practice']).values
print('label tensor shape: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')

matrix = np.zeros((vocab_size, 300))
for word, i in vocab.items():
    if i >= vocab_size:
        continue

    if word in vec.wv.vocab:
        vector = vec[word]

        if vector is not None and len(vector) > 0:
            matrix[i] = vector

model = cnn(matrix, vocab)
#model = lstm(matrix, vocab)
model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=2)

y_pred = model.predict(x_test)
y_test, y_pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred))

# select top 1000 features
#descending = sorted(corpus.iindex.index, key=lambda x: [node['freq'] for node in corpus.iindex.index[x] if node['id'] == -1], reverse=True)
#features = descending[:1000]
#opp115['segment'] = opp115['segment'].apply(lambda x: ' '.join([word for word in x.split() if word in features]))

#statistics = corpus.generate_statistics()
# print(statistics)

#cnn = CNN(opp115)
#cnn.evaluate(num_filters=200, ngrams=[2, 3], save_results=False)

#svm = SVM(opp115)
#svm.evaluate(alpha=0.0001, iterations=100, tolerance=0.001, save_results=False)

#mnb = MNB(opp115)
#mnb.evaluate(alpha=0.1, save_results=False)