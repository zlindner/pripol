import numpy as np
from data.acl1010 import ACL1010
from data.opp115 import OPP115
from keras import layers
from keras.models import Model

class CNN:

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

	# creates an embedding layer for the given word vectors
	def build_embedding_layer(self, vectors, max_seq_len):
		matrix = self.build_embedding_matrix(vectors)

		embed = layers.Embedding(
			input_dim=matrix.shape[0],
			output_dim=matrix.shape[1],
			weights=[matrix],
			input_length=max_seq_len,
			trainable=False
		)

		return embed

	# builds the cnn architecture
	def build(self, max_seq_len):
		embed_acl1010 = self.build_embedding_layer(ACL1010().load_vectors(), max_seq_len)

		seq_input = layers.Input(shape=(max_seq_len,), dtype='int32')
		embed_seq = embed_acl1010(seq_input)

		convs = []
		filter_sizes = [3, 4, 5]
	
		for filter_size in filter_sizes:
			conv = layers.Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embed_seq)
			pool = layers.MaxPooling1D(pool_size=3)(conv)
			convs.append(pool)

		conv = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(embed_seq)
		pool = layers.MaxPooling1D(pool_size=3)(conv)

		x = layers.Dropout(0.5)(pool)
		x = layers.Flatten()(x)
		x = layers.Dense(128, activation='relu')(x)
		x = layers.Dropout(0.5)(x)

		output = layers.Dense(len(OPP115.DATA_PRACTICES), activation='sigmoid')(x)

		model = Model(seq_input, output)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
		
		print(model.summary())

		return model
		
# partitions X into training and validation subsets n times
def split_stratified_kfold(X, y, n):
	skf = StratifiedKFold(n_splits=n, random_state=42)
	
	for train_index, val_index in skf.split(X, y):
		X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(val_index)]
		y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(val_index)]

		encode_sequences(X_train, X_val)