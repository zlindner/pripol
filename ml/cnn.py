import data.vectors as vectors
import numpy as np
import time

from data.corpus import Corpus
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense, Concatenate, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report


class CNN():

    def __init__(self, opp115):
        self.corpus = opp115

        self.embedding_dim = 300  # embedding dimension size
        self.max_len = 100  # max sequence length

        self.vectors = vectors.load_static('acl')

    def create(self, vocab, num_filters, ngrams):
        vocab_size = len(vocab) + 1
        matrix = self.create_matrix(vocab, vocab_size, self.vectors)

        model = Sequential()

        model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=self.max_len, trainable=False))

        convs = []

        for ngram in ngrams:
            conv = Conv1D(num_filters, ngram, activation='relu')
            convs.append(conv)

        if len(ngrams) > 1:
            model.add(Concatenate()(convs))
        else:
            model.add(convs[0])

        # model.add(Conv1D(num_filters, ngram_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(Corpus.DATA_PRACTICES), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_functional(self, vocab, num_filters, ngrams):
        vocab_size = len(vocab) + 1
        matrix = self.create_matrix(vocab, vocab_size, self.vectors)

        i = Input(shape=(self.max_len,))
        embed = Embedding(vocab_size, 300, weights=[matrix], input_length=self.max_len, trainable=False)(i)

        convs = []

        for ngram in ngrams:
            layer = Conv1D(num_filters, ngram, activation='relu')(embed)
            layer = GlobalMaxPooling1D()(layer)
            # conv = Flatten()(conv)
            convs.append(layer)

        layers = Concatenate()(convs) if len(convs) > 1 else convs[0]
        # pooling = GlobalMaxPooling1D()(conv_layers)
        drop = Dropout(0.5)(layers)
        d = Dense(100, activation='relu')(drop)
        output = Dense(len(Corpus.DATA_PRACTICES), activation='softmax')(d)

        model = Model(i, output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_matrix(self, vocab, vocab_size, vectors):
        matrix = np.zeros((vocab_size, self.embedding_dim))

        for word, i in vocab.items():
            if i >= vocab_size:
                continue

            if word in vectors.wv.vocab:
                vector = vectors[word]

                if vector is not None and len(vector) > 0:
                    matrix[i] = vector

        return matrix

    def create_sequences(self, x_train, x_test):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        x_train = pad_sequences(x_train, self.max_len, padding='post')
        x_test = pad_sequences(x_test, self.max_len, padding='post')

        vocab = tokenizer.word_index

        return x_train, x_test, vocab

    def evaluate(self, num_filters, ngrams, save_results=False):
        x = self.corpus['segment']
        y = self.corpus[Corpus.DATA_PRACTICES].values

        true, pred, times = self.kfold(x, y, num_filters, ngrams, folds=10, epochs=20)
        results = classification_report(true, pred, target_names=Corpus.DATA_PRACTICES)  # accuracy = micro avg

        print(results)
        print('Total time elapsed: %ss\n' % np.round(sum(times), 2))

        if save_results:
            with open('results/cnn_results.txt', 'a') as f:
                f.write('num_filters: %s, ngram_size: %s\n%s\n\nTotal time elapsed: %ss\n\n' % (num_filters, ngrams, results, sum(times)))

    def tune(self):
        grid = {
            'num_filters': [100, 200, 300, 400, 500, 600, 700],
            'ngram_size': [3, 4, 5, 6, 7]
        }

        param_grid = list(ParameterGrid(grid))

        for param in param_grid:
            print(param)

            x = self.corpus['segment']
            y = self.corpus[Corpus.DATA_PRACTICES].values

            true, pred, times = self.kfold(x, y, param['num_filters'], param['ngram_size'], folds=10, epochs=20)

            results = classification_report(true, pred, target_names=Corpus.DATA_PRACTICES, output_dict=True)
            param['accuracy'] = results['accuracy']
            param['time'] = times

            print(param['accuracy'])

        best = max(param_grid, key=lambda x: x['accuracy'])

        print(param_grid)
        print(best)

    def kfold(self, x, y, num_filters, ngrams, folds, epochs):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        fold = 1  # current fold
        true = []
        pred = []
        times = []  # list to elapsed time for each fold

        for train, test in skf.split(x, np.argmax(y, axis=1)):
            print('Fold #%s...' % fold, end='', flush=True)
            fold += 1
            fold_start = time.time()

            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]

            x_train, x_test, vocab = self.create_sequences(x_train, x_test)

            # model = self.create(vocab, num_filters, ngrams)
            model = self.create_functional(vocab, num_filters, ngrams)
            model.fit(x_train, y_train, epochs=epochs, batch_size=50, verbose=0)

            y_pred = model.predict(x_test)
            y_test, y_pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)

            true.extend(y_test)
            pred.extend(y_pred)

            fold_end = time.time()
            elapsed = round(fold_end - fold_start, 2)
            times.append(elapsed)
            print('%ss' % elapsed)  # elapsed time for each fold

        return true, pred, times
