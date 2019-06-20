import numpy as np

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_validate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras import layers

class CNN():

	def __init__(self, opp115, vectors):
		print('\n# CNN\n')

		# embedding hyperparameters
		self.embedding_dim = 300
		self.max_len = 100

		# convolution layer hyperparameters
		self.filters = 200

		# training hyperparameters
		self.epochs = 10
		self.batch_size = 40
		
		self.labels = opp115.labels
		
		# split into train and test subsets
		df = opp115.consolidated.merge(opp115.encoded, on=['policy_id', 'segment_id']).drop_duplicates()
		x_train, x_test, y_train, y_test = self.initial_split(df)

		# tokenize and create sequences
		x_train, x_test = self.tokenize(x_train, x_test)
		y_test = np.argmax(y_test, axis=1)

		self.matrix = self.create_matrix(vectors)
		#self.tune_hyperparameters(x_train, x_test, y_train, y_test)

		# create the model
		self.model = self.create_model(self.filters, 3)

		self.metrics = {
			'precision': make_scorer(precision_score, average='micro'),
			'recall': make_scorer(recall_score, average='micro'),
			'f1': make_scorer(f1_score, average='micro'),
			'tp': make_scorer(self.tp),
			'tn': make_scorer(self.tn),
			'fp': make_scorer(self.fp),
			'fn': make_scorer(self.fn)
		}

		y_train = np.argmax(y_train, axis=1)
		kc = KerasClassifier(build_fn=self.create_model, epochs=10, num_filters=self.filters, kernel_size=3)
		scores = cross_validate(estimator=kc, X=x_train, y=y_train, cv=10, scoring=self.metrics)

		print(scores)

		return

		# train the model
		self.train(x_train, y_train)

		# predict
		y_pred = self.model.predict(x_test)
		y_pred = np.argmax(y_pred, axis=1)

		print(classification_report(y_test, y_pred, target_names=self.labels))

	# creates the cnn architecture
	def create_model(self, num_filters, kernel_size):
		model = Sequential()

		# embedding layer trainable???
		model.add(layers.Embedding(self.vocab_size, self.embedding_dim, weights=[self.matrix], input_length=self.max_len, trainable=False))
		model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
		model.add(layers.GlobalMaxPooling1D())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(100, activation='relu'))
		model.add(layers.Dense(12, activation='softmax'))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	# creates cnn architecture using multiple convolutional layers with varying kernel sizes
	def create_model_2(self, matrix):
		i = layers.Input(shape=(self.max_len,))
		e = layers.Embedding(self.vocab_size, self.embedding_dim, weights=[matrix], input_length=self.max_len, trainable=False)(i)

		convs = []
		ngrams = [3, 4, 5]

		for ngram in ngrams:
			c = layers.Conv1D(self.filters, ngram, activation='relu')(e)
			c = layers.MaxPooling1D()(c)
			c = layers.Flatten()(c)
			convs.append(c)

		c = layers.Concatenate()(convs)
		d = layers.Dropout(0.5)(c)
		d = layers.Dense(100, activation='relu')(d)
		d = layers.Dense(12, activation='softmax')(d)

		model = Model(i, d)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	# trains the cnn using stratified kfold cross validation
	def train(self, x, y):
		print('training...')

		skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
		fold = 1

		for t, v in skf.split(x, np.argmax(y, axis=1)):
			print('fold %s...' % fold, end='', flush=True)

			x_train, x_val = x[t], x[v]
			y_train, y_val = y[t], y[v]

			self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_val, y_val), verbose=0)

			print('done!')
			fold += 1

	# creates an embedding matrix from word vectors
	def create_matrix(self, vectors):
		print('creating embedding matrix...')

		matrix = np.zeros((self.vocab_size, self.embedding_dim))

		for word, i in self.vocab.items():
			if i >= self.vocab_size:
				continue

			if word in vectors.wv.vocab:
				vector = vectors[word]

				if vector is not None and len(vector) > 0:
					matrix[i] = vector	

		return matrix

	def tune_hyperparameters(self, x_train, x_test, y_train, y_test):
		self.params = {
			'num_filters': [100, 200, 300, 400, 500, 600],
			'kernel_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		}

		self.metrics = {
			'precision': make_scorer(precision_score, average='micro'),
			'recall': make_scorer(recall_score, average='micro'),
			'f1': make_scorer(f1_score, average='micro'),
			'tp': make_scorer(self.tp),
			'tn': make_scorer(self.tn),
			'fp': make_scorer(self.fp),
			'fn': make_scorer(self.fn)
		}

		skf = StratifiedKFold(n_splits=10, random_state=42)

		model = KerasClassifier(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
		rand = GridSearchCV(model, self.params, cv=skf)

		rand.fit(x_train, np.argmax(y_train, axis=1))

		print('best parameters set found on development set')
		print(rand.best_params_)

		test_acc = rand.score(x_test, y_test)
		print(test_acc)

	# initial split into testing and training / validation subsets
	def initial_split(self, df):
		x = df['segment']
		y = df[self.labels].values
		
		return train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

	# TODO look into setting max number of words for feature selection
	# tokenize and create text sequences
	def tokenize(self, x_train, x_test, max_features=None):
		tokenizer = Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(x_train)

		x_train = tokenizer.texts_to_sequences(x_train)
		x_test = tokenizer.texts_to_sequences(x_test)

		x_train = pad_sequences(x_train, self.max_len, padding='post')
		x_test = pad_sequences(x_test, self.max_len, padding='post')

		self.vocab = tokenizer.word_index
		self.vocab_size = len(self.vocab) + 1 # add 1 due to reserved 0 index

		return x_train, x_test

	# true positive metric
	def tp(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[1, 1]

	# true negative metric
	def tn(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[0, 0]

	# false positive metric
	def fp(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[0, 1]
	
	# false negative metric
	def fn(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0
		
		return matrix[1, 0]
