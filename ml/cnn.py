import numpy as np
from ml.model import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense


class CNN(Model):

    def __init__(self, data, vectors):
        print('\n# CNN\n')

        self.max_features = None    # max # of features to keep when tokenizing
        self.max_len = 100 			# max sequence length
        self.embedding_dim = 300 	# embedding dimension size
        self.num_filters = 100		# convolutional layer filter size
        self.ngram_size = 3			# convolutional layer ngram size
        self.dense_size = 100		# dense layer size
        self.epochs = 10			# number of epochs when training
        self.batch_size = 40		# batch size when training

        self.vectors = vectors
        self.binary = False

        x = data['segment']
        y = np.argmax(data[Model.LABELS].values, axis=1)

        super().__init__(x, y)

    # wrapper for create_model TODO change to def create(self, kwargs)
    def create(self):
        matrix = self.create_matrix(self.vectors)

        return KerasClassifier(build_fn=self.create_model, matrix=matrix, max_len=self.max_len, num_filters=self.num_filters, ngram_size=self.ngram_size, dense_size=self.dense_size, epochs=self.epochs)

    def create_model(self, matrix, max_len, num_filters, ngram_size, dense_size):
        model = Sequential()

        model.add(Embedding(self.vocab_size, self.embedding_dim, weights=[matrix], input_length=max_len))
        model.add(Conv1D(num_filters, ngram_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(len(Model.LABELS), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_test_set(self, x, y):
        x_train, x_test, y_train, y_test = super().create_test_set(x, y)
        x_train, x_test = self.create_sequences(x_train, x_test)

        return x_train, x_test, y_train, y_test

    def create_matrix(self, vectors):
        print('creating embedding matrix...', end='', flush=True)

        matrix = np.zeros((self.vocab_size, self.embedding_dim))

        for word, i in self.vocab.items():
            if i >= self.vocab_size:
                continue

            if word in vectors.wv.vocab:
                vector = vectors[word]

                if vector is not None and len(vector) > 0:
                    matrix[i] = vector

        print('done!')

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
