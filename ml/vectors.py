import numpy as np
from gensim.models import KeyedVectors as kv

# loads google news vectors
def load_google():
	print('\n# loading google news vectors')
	vectors = kv.load_word2vec_format('data/google/google-news.bin', binary=True)
	print('loaded %s vectors' % (len(vectors.vocab)))
	return vectors

load_google()