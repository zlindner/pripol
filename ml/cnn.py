import numpy as np
from ml.model import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense


class CNN(Model):

    def __init__(self, vectors):
        print('\n# CNN\n')

        self.vectors = vectors
        self.binary = False
        self.max_features = None    # max # of features to keep when tokenizing
        self.max_len = 100          # max sequence length
        self.embedding_dim = 300 	# embedding dimension size
        self.epochs = 10			# number of epochs when training
        self.batch_size = 40		# training batch size

        self.model = KerasClassifier(build_fn=self.create, epochs=self.epochs, batch_size=self.batch_size)

    def create(self, num_filters=600, ngram_size=6, dense_size=100):
        print('params: %s %s' % (num_filters, ngram_size))

        matrix = self.create_matrix(self.vectors)

        model = Sequential()

        model.add(Embedding(self.vocab_size, self.embedding_dim, weights=[matrix], input_length=self.max_len, trainable=False))
        model.add(Conv1D(num_filters, ngram_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(len(Model.LABELS), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_matrix(self, vectors):
        matrix = np.zeros((self.vocab_size, self.embedding_dim))

        for word, i in self.vocab.items():
            if i >= self.vocab_size:
                continue

            if word in vectors.wv.vocab:
                vector = vectors[word]

                if vector is not None and len(vector) > 0:
                    matrix[i] = vector

        return matrix

    def create_sequences(self, x_train, x_test):
        print('creating sequences...', end='', flush=True)

        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        x_train = pad_sequences(x_train, self.max_len, padding='post')
        x_test = pad_sequences(x_test, self.max_len, padding='post')

        self.vocab = tokenizer.word_index
        self.vocab_size = len(self.vocab) + 1

        print('done!')

        return x_train, x_test

    # TODO is this allowed?
    def create_sequences(self, x):
        print('creating sequences...', end='', flush=True)

        tokenizer = Tokenizer(num_words=self.max_features)
        tokenizer.fit_on_texts(x)

        x = tokenizer.texts_to_sequences(x)

        x = pad_sequences(x, self.max_len, padding='post')

        self.vocab = tokenizer.word_index
        self.vocab_size = len(self.vocab) + 1

        print('done!')

        return x
