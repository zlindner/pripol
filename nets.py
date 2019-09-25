from opp115 import DATA_PRACTICES
from keras.models import Sequential 
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Dense

def cnn(matrix, vocab):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(Conv1D(100, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def lstm(matrix, vocab):
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, weights=[matrix], input_length=200, trainable=False))
    model.add(LSTM(150, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dense(len(DATA_PRACTICES), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model