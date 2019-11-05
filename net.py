import numpy as np
from opp115 import DATA_PRACTICES
from keras.models import Sequential 
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Dense, Bidirectional, MaxPooling1D,
from keras.optimizers import Adam, Nadam, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.callbacks.callbacks import EarlyStopping
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report
from time import time

def cnn(matrix, vocab_size, params):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=100, trainable=False))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstm(matrix, vocab_size, params):
    model = Sequential()

    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=100, trainable=False))
    model.add(Dropout(params['embedding_dropout']))     

    for layer in range(params['lstm_layers'] - 1):
        model.add(LSTM(params['memory_dim'], return_sequences=True, dropout=params['lstm_dropout'], recurrent_dropout=params['recurrent_dropout']))

    model.add(LSTM(params['memory_dim'], dropout=params['lstm_dropout'], recurrent_dropout=params['recurrent_dropout']))

    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))

    if params['optimizer'] == 'adam':
        optimizer = Adam(clipnorm=1)
    elif params['optimizer'] == 'adam_0.01':
        optimizer = Adam(lr=0.01, clipnorm=1)
    elif params['optimizer'] == 'nadam':
        optimizer = Nadam(clipnorm=1)        
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(clipnorm=1)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
 
    return model

def bi_lstm(matrix, vocab_size, params):
    model = Sequential()

    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=100, trainable=False))
    model.add(Dropout(params['embedding_dropout']))     

    for layer in range(params['lstm_layers'] - 1):
        model.add(Bidirectional(LSTM(params['memory_dim'], return_sequences=True, dropout=params['lstm_dropout'], recurrent_dropout=params['recurrent_dropout'])))

    model.add(Bidirectional(LSTM(params['memory_dim'], dropout=params['lstm_dropout'], recurrent_dropout=params['recurrent_dropout'])))

    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))

    if params['optimizer'] == 'adam':
        optimizer = Adam(clipnorm=1)
    elif params['optimizer'] == 'adam_0.01':
        optimizer = Adam(lr=0.01, clipnorm=1)
    elif params['optimizer'] == 'nadam':
        optimizer = Nadam(clipnorm=1)        
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(clipnorm=1)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
 
    return model

def cnn_lstm(matrix, vocab_size, params):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=100, trainable=False))

    if params['embedding_dropout'] is not None:
        model.add(Dropout(params['embedding_dropout']))

    model.add(Conv1D(100, 3, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.5))

    model.add(LSTM(params['memory_dim']))
    
    if params['lstm_dropout'] is not None:
        model.add(Dropout(params['lstm_dropout']))

    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# TODO lstm svm

def init_sequences(x, padding):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    #print('found %s unique tokens' % len(tokenizer.word_index))

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
    if model_name == 'cnn':
        x, vocab = init_sequences(x, padding='post')
    else: 
        x, vocab = init_sequences(x, padding='pre') # lstm variants
    
    vocab_size = len(vocab) + 1
    vec = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    #vec = KeyedVectors.load_word2vec_format('google-news.bin', binary=True)
    matrix = init_matrix(vocab, vec)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_num = 1
    actual = []
    predictions = []
    total_time = 0
    total_epochs = 0

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
        elif model_name == 'bi_lstm':
            model = bi_lstm(matrix, vocab_size, params)
        elif model_name == 'cnn_lstm':
            model = cnn_lstm(matrix, vocab_size, params)
        elif model_name == 'stacked_lstm':
            model = stacked_lstm(matrix, vocab_size, params)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=0, callbacks=[early_stopping])
        fold_epochs = early_stopping.stopped_epoch
        total_epochs += fold_epochs

        y_predict = model.predict(x_test)
        y_test, y_predict = np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)

        actual.extend(y_test)
        predictions.extend(y_predict)

        fold_end = time()
        fold_time = round(fold_end - fold_start, 2)
        total_time += fold_time

        if verbose:
            print('\t\t%s epochs\t\t%ss' % (fold_epochs, fold_time))

    results = classification_report(actual, predictions, output_dict=True)
    avg_epochs = total_epochs / 10
    
    if verbose:
        print(classification_report(actual, predictions))
        print('average epochs: %s' % avg_epochs)

    return results, round(total_time, 2), avg_epochs

def tune(x, y, model_name, params):
    grid = list(ParameterGrid(params))

    print('# tuning %s with parameters %s #\n' % (model_name, params))

    for params in grid:
        print('testing %s' % params, end='', flush=True)
        
        results, eval_time, avg_epochs = evaluate(x, y, model_name, params, verbose=False)
        f = round(results['accuracy'], 2)
        
        print('\t\t%s\t\t%s epochs\t\t%ss' % (f, avg_epochs, eval_time))

def get_layer_output(n, cnn, x):
    get_layer_output = K.function([cnn.layers[0].input], [cnn.layers[n].output])
    return get_layer_output([x])[0]