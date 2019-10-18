import numpy as np
from opp115 import DATA_PRACTICES
from keras.models import Sequential 
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Dense, ConvLSTM2D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from time import time

def cnn(matrix, vocab_size, params):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstm(matrix, vocab_size, params):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=100, trainable=False))
    #model.add(Dropout(0.5))
    model.add(LSTM(params['memory_dim']))
    #model.add(Dropout(0.5))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def cnn_lstm(matrix, vocab_size, params):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(ConvLSTM2D(filters=100, kernel_size=3))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def init_sequences(x, padding):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    print('found %s unique tokens' % len(tokenizer.word_index))

    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=100, padding=padding)
    
    return x, vocab

def init_matrix(vocab, vec):
    vocab_size = len(vocab) + 1
    matrix = np.zeros((vocab_size, 300))

    for word, i in vocab.items():
        if i >= vocab_size:
            continue

        if word in vec.wv.vocab:
            vector = vec[word]

            if vector is not None and len(vector) > 0:
                matrix[i] = vector

    return matrix

def evaluate(x, y, model_name, params, verbose=True):
    if model_name == 'lstm' or model_name == 'cnn_lstm': # TODO not sure if cnn_lstm should use pre or post padding
        x, vocab = init_sequences(x, padding='post')
    elif model_name == 'cnn':
        x, vocab = init_sequences(x, padding='pre')
    
    vocab_size = len(vocab) + 1
    vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    matrix = init_matrix(vocab, vec)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_num = 1
    actual = []
    predictions = []
    total_time = 0

    for train, test in skf.split(x, np.argmax(y, axis=1)):
        if verbose:
            print('fold %s' % fold_num, end='', flush=True)

        fold_num += 1
        fold_start = time()

        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        if model_name == 'lstm':
            model = lstm(matrix, vocab_size, params)
        elif model_name == 'cnn':
            model = cnn(matrix, vocab_size, params)
        elif model_name == 'cnn_lstm':
            model = cnn_lstm(matrix, vocab_size, params)

        model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=0)

        y_predict = model.predict(x_test)
        y_test, y_predict = np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)

        actual.extend(y_test)
        predictions.extend(y_predict)

        fold_end = time()
        fold_time = round(fold_end - fold_start, 2)
        total_time += fold_time

        if verbose:
            print('\t\t%ss' % fold_time)

    results = classification_report(actual, predictions, output_dict=True)

    if verbose:
        print(classification_report(actual, predictions))

    return results, round(total_time, 2)

def get_layer_output(n, cnn, x):
    get_layer_output = K.function([cnn.layers[0].input], [cnn.layers[n].output])
    return get_layer_output([x])[0]