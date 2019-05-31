import numpy as np
from gensim.models import KeyedVectors as kv

# loads acl1010 vectors
def load_acl1010():
	print('\n# loading acl1010 vectors')
	vectors = kv.load('../data/acl1010/acl1010.vec', mmap='r')
	print('loaded %s vectors' % (len(vectors.wv.vocab)))
	return vectors

# loads google news vectors
def load_google():
	print('\n# loading google news vectors')
	vectors = kv.load_word2vec_format('../data/google/google-news.bin', binary=True)
	print('loaded %s vectors' % (len(vectors.vocab)))
	return vectors

# builds an embedding matrix from word2vec
def build_embedding_matrix(vectors, dim):
	vocab = vectors.wv.vocab
	vocab_size = len(vectors.wv.vocab) + 1

	matrix = np.zeros((vocab_size, dim))

	for i, word in enumerate(vocab):
		vec = vectors[word]

		if vec is not None:
			matrix[i] = vec

	return matrix