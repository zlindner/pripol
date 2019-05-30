import vectors
from keras.models import Model
from keras import layers
from keras.activations import relu

# TODO Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMAs
# - should I remove , . from text?
# - does each embedding require its own embedding layer? -> acl1010 and google
# - does each n-gram require its own embedding, max-pooling layers?
# - other statistics to generate?

vec_acl1010 = vectors.load_acl1010()
#vec_google = vectors.load_google()

# builds an embedding layer for the given embedding matrix
def build_embedding_layer(matrix, max_len, trainable=False):
	print('building embedding layer')

	embed_layer = layers.Embedding(
		input_dim=matrix.shape[0],
		output_dim=matrix.shape[1],
		input_length=max_len,
		weights=[matrix],
		trainable=trainable,
	)

	return embed_layer

def build_convolutional_pool(x_input, max_len, n_grams=[3, 4, 5], feature_maps=300):
	print('building convolutional pool...')

	pool = []

	for n in n_grams:
		branch = layers.Conv1D(filters=feature_maps, kernel_size=n, activation=relu)(x_input)
		branch = layers.MaxPooling1D(pool_size=max_len - n + 1, strides=None, padding='valid')(branch)
		branch = layers.Flatten()(branch)

		pool.append(branch)

	return pool

# builds the convolutional neural network architecture
# TODO get input_length for Embedding layer... padding?
def build_cnn():
	print('building cnn architecture...')

	matrix_acl1010 = vectors.build_embedding_matrix(vec_acl1010, 300)
	embed_acl1010 = build_embedding_layer(matrix_acl1010, 10)

	i = layers.Input(shape=(10,), dtype='int32')
	x = embed_acl1010(i)

	conv_pool = build_convolutional_pool(x, 10)
	z = layers.concatenate(conv_pool, axis=-1)

	o = layers.Dense(1, activation='sigmoid')(z)
	
	model = Model(inputs=i, outputs=o)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	print(model.summary())

	return model

model = build_cnn()	