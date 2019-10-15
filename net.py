import numpy as np
from opp115 import DATA_PRACTICES
from keras.models import Sequential 
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

def cnn(matrix, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstm(matrix, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(LSTM(200))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def init_sequences(x, padding):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index
    print('found %s unique tokens' % len(tokenizer.word_index))

    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=200, padding=padding)
    
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

def get_layer_output(n, cnn, x):
    get_layer_output = K.function([cnn.layers[0].input], [cnn.layers[n].output])
    return get_layer_output([x])[0]