import data.vectors as vectors
import numpy as np
import time

from data.corpus import Corpus
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report


class CNN():

    def __init__(self, opp115):
        self.corpus = opp115

        self.embedding_dim = 300  # embedding dimension size
        self.max_len = 100  # max sequence length

        self.vectors = vectors.load_static('acl')

    def create(self, vocab, num_filters, ngram_size):
        vocab_size = len(vocab) + 1
        matrix = self.create_matrix(vocab, vocab_size, self.vectors)

        model = Sequential()

        model.add(Embedding(vocab_size, 300, weights=[matrix], input_length=self.max_len, trainable=False))
        model.add(Conv1D(num_filters, ngram_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(Corpus.DATA_PRACTICES), activation='softmax'))

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

    def evaluate(self, num_filters, ngram_size, output_file=None):
        x = self.corpus['segment']
        y = self.corpus[Corpus.DATA_PRACTICES].values

        true, pred, times = self.kfold(x, y, num_filters, ngram_size, folds=3, epochs=1)
        results = classification_report(true, pred, target_names=Corpus.DATA_PRACTICES)  # accuracy = micro avg

        print(results)
        print('Total time elapsed: %ss\n' % np.round(sum(times), 2))

        if output_file is not None:
            with open('results/' + output_file, 'w') as f:
                f.write(results + '\nTotal time elapsed: %ss\n' % sum(times))

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

    def kfold(self, x, y, num_filters, ngram_size, folds, epochs):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        fold = 1
        true = []
        pred = []
        times = []

        for train, test in skf.split(x, np.argmax(y, axis=1)):
            print('Fold # %s...' % fold, end='', flush=True)
            fold += 1
            fold_start = time.time()

            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]

            x_train, x_test, vocab = self.create_sequences(x_train, x_test)

            model = self.create(vocab, num_filters, ngram_size)
            model.fit(x_train, y_train, epochs=epochs, batch_size=50, verbose=0)

            y_pred = model.predict(x_test)
            y_test, y_pred = np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)

            true.extend(y_test)
            pred.extend(y_pred)

            fold_end = time.time()
            elapsed = np.round(fold_end - fold_start, 2)
            times.append(elapsed)
            print('%ss' % elapsed)  # elapsed time for each fold

        return true, pred, times
