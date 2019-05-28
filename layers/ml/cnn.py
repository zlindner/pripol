import vectors
from keras.models import Sequential
from keras import layers

# TODO Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMAs

vec_acl1010 = vectors.load_acl1010()
#vec_google = vectors.load_google()

# builds the convolutional neural network architecture
# TODO get input_length for Embedding layer
def build_cnn():
	print('building cnn architecture...')

	embed_dim = 300

	matrix_acl1010 = vectors.build_embedding_matrix(vec_acl1010, embed_dim)
	embed_acl1010 = layers.Embedding(len(matrix_acl1010), embed_dim, weights=[matrix_acl1010], input_length=10, trainable=False)
	
	#embed_google = layers.Embedding(vec_google.wv.vocab, embed_dim, trainable=False)
	#embed_matrix = layers.Add([embed_acl1010, embed_google])

	model = Sequential()
	model.add(embed_acl1010)
	#model.add(layers.Conv1D()) TODO convolutional layer

	model.compile(optimizer='adam', loss='binary_crossentropy')

	print(model.summary())


build_cnn()	