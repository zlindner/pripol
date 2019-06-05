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

	# creates and embedding matrix for the given word vectors
	def build_embedding_matrix(self, vectors, dimension = 300):
		vocab = vectors.wv.vocab
		vocab_size = len(vocab) + 1

		matrix = np.zeros((vocab_size, dimension))

		for i, word in enumerate(vocab):
			vec = vectors[word]

			if vec is not None:
				matrix[i] = vec
		
		return matrix

	# TODO not sure if this is right
	def build_embedding_matrix(self, vectors, vocab, vocab_size, dimension = 300):
		matrix = np.zeros((vocab_size, dimension))

		for i, word in enumerate(vectors.wv.vocab):
			vec = vectors[word]

			if word in vocab:
				idx = vocab[word]
				matrix[idx] = np.array(vec, dtype=np.float32)[:dimension]
			
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
		seq_input = layers.Input(shape=(self.opp115.max_seq_len,), dtype='int32')

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
			pool = layers.MaxPooling1D(pool_size=self.opp115.max_seq_len - ngram + 1)(conv)
			flat = layers.Flatten()(pool)
			# flatten?
			convs.append(flat)

		merge = layers.Concatenate(axis=1)(convs)

		#dropout = layers.Dropout(0.5)(merge)
		#flatten = layers.Flatten()(merge)
		dense = layers.Dense(128, activation='relu')(merge)
		dropout = layers.Dropout(0.5)(dense)
		dense = layers.Dense(10, activation='softmax')(dropout) # sigmoid or softmax?

		model = Model(seq_input, dense)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) # binary or categorical crossentropy?
		
		return model
		
# partitions X into training and validation subsets n times
def split_stratified_kfold(X, y, n):
	skf = StratifiedKFold(n_splits=n, random_state=42)
	
	for train_index, val_index in skf.split(X, y):
		X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(val_index)]
		y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(val_index)]

		encode_sequences(X_train, X_val)