import numpy as np
from data.acl1010 import ACL1010
from data.opp115 import OPP115
from gensim.models import KeyedVectors as kv
from keras import layers
from keras.models import Model

class CNN:

	def __init__(self, opp115, acl1010):
		self.opp115 = opp115
		self.acl1010 = acl1010

	# 
	def build_embedding_matrix(self, vectors, vocab, vocab_size, dimension = 300):
		# not sure which is better
		matrix = np.zeros((vocab_size, dimension)) 
		#matrix = np.random.rand(vocab_size, dimension)

		for word, i in vocab.items():
			if i >= vocab_size:
				continue

			if word in vectors.wv.vocab:
				vector = vectors[word]

				if vector is not None and len(vector) > 0:
					matrix[i] = vector

		return matrix

	# creates an embedding layer for the given word vectors
	def build_embedding_layer(self, vectors):
		matrix = self.build_embedding_matrix(vectors, self.opp115.vocab, self.opp115.vocab_size)

		embed = layers.Embedding(
			input_dim=matrix.shape[0],
			output_dim=matrix.shape[1],
			weights=[matrix],
			input_length=self.opp115.max_seq_len,
			trainable=False
		)

		return embed

	# loads the google news word vectors
	def load_google_vectors(self):
		print('\n# loading word vectors')

		vectors = kv.load_word2vec_format('data/google/google-news.bin', binary=True)
		print('loaded %s vectors' % (len(vectors.wv.vocab)))

		return vectors

	# builds the cnn architecture
	def build(self):
		seq_input = layers.Input(shape=(self.opp115.max_seq_len,))

		# acl1010 embedding layer
		vec_acl1010 = self.acl1010.load_vectors()
		embed_acl1010 = self.build_embedding_layer(vec_acl1010)
		embed_seq_acl1010 = embed_acl1010(seq_input)

		# google news embedding layer
		#vec_google = self.load_google_vectors()
		#embed_google = self.build_embedding_layer(vec_google)
		#embed_seq_google = embed_google(seq_input)
		
		convs = []
		ngram_sizes = [3, 4, 5]

		for ngram in ngram_sizes:
			conv = layers.Conv1D(filters=100, kernel_size=ngram, activation='relu')(embed_seq_acl1010)
			#pool = layers.MaxPooling1D(pool_size=self.opp115.max_seq_len - ngram + 1)(conv) # results in lower accuracy
			pool = layers.MaxPooling1D(pool_size=2)(conv) # pool size?
			flat = layers.Flatten()(pool)
			convs.append(flat)

		merge = layers.Concatenate()(convs)

		dropout = layers.Dropout(0.5)(merge)
		dense = layers.Dense(100, activation='relu')(dropout) # 128? polisis says 100
		dense = layers.Dense(10, activation='softmax')(dense)

		model = Model(seq_input, dense)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # polisis says categorical_crossentropy

		print(model.summary())
		
		return model
		
# partitions X into training and validation subsets n times
def split_stratified_kfold(X, y, n):
	skf = StratifiedKFold(n_splits=n, random_state=42)
	
	for train_index, val_index in skf.split(X, y):
		X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(val_index)]
		y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(val_index)]

		encode_sequences(X_train, X_val)