import numpy as np

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
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

		# TODO move to opp115.py
		self.labels = [
			'first_party_collection_use',
			'third_party_sharing_collection', 
			'introductory_generic', 
			'user_choice_control', 
			'international_specific_audiences', 
			'data_security', 
			'privacy_contact_information', 
			'user_access_edit_deletion', 
			'practice_not_covered', 
			'policy_change', 
			'data_retention', 
			'do_not_track'
		]
		
		# split into train and test subsets
		df = opp115.consolidated.merge(opp115.encoded, on=['policy_id', 'segment_id']).drop_duplicates()
		x_train, x_test, y_train, y_test = self.initial_split(df)

		# tokenize and create sequences
		x_train, x_test = self.tokenize(x_train, x_test)
		y_test = np.argmax(y_test, axis=1)

		matrix = self.create_matrix(vectors)
		self.model = self.create_model(matrix)

		self.train(x_train, y_train)

		y_pred = self.model.predict(x_test)
		y_pred = np.argmax(y_pred, axis=1)

		print(classification_report(y_test, y_pred, target_names=self.labels))

	# creates the cnn architecture
	def create_model(self, matrix):
		model = Sequential()

		# embedding layer trainable???
		model.add(layers.Embedding(self.vocab_size, self.embedding_dim, weights=[matrix], input_length=self.max_len, trainable=False))
		model.add(layers.Conv1D(self.filters, 3, activation='relu'))
		model.add(layers.GlobalMaxPooling1D())
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(100, activation='relu'))
		model.add(layers.Dense(12, activation='softmax'))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		return model

	# trains the cnn using stratified kfold cross validation
	def train(self, x, y):
		print('training...')

		skf = StratifiedKFold(n_splits=10, random_state=42)
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