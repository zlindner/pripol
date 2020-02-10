import opp115
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

class CNN():

    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.x, self.y, self.matrix = self.load_dataset()
        self.model = self.create()
        self.train()

    def load_embeddings(self):
        print('Loading word embeddings...', end='', flush=True)
        embeddings = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
        print('done!')
        
        return embeddings

    def load_dataset(self):
        data = opp115.load()
        data = data[data['data_practice'] != 'Other']

        x, vocab = self.init_sequences(data['text'].values)
        matrix = self.init_matrix(vocab, self.embeddings)

        return x, pd.get_dummies(data['data_practice']).values, matrix

    def init_sequences(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        vocab = tokenizer.word_index

        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=100, padding='post')
        
        return x, vocab

    def init_matrix(self, vocab, vec):
        self.vocab_size = len(vocab) + 1
        matrix = np.zeros((self.vocab_size, 300))

        for word, i in vocab.items():
            if i >= self.vocab_size:
                continue

            if word in vec.wv.vocab:
                vector = vec[word]

                if vector is not None and len(vector) > 0:
                    matrix[i] = vector

        return matrix

    def create(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 300, weights=[self.matrix], input_length=100, trainable=False))
        model.add(Conv1D(100, 3, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(len(opp115.DATA_PRACTICES), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    
    def train(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)
        self.model.fit(self.x, self.y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
        self.model.save('models/cnn.h5')
