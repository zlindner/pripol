from gensim.models import KeyedVectors as kv
import acl1010

# load google news vectors
def load_google_news():
    print('\n# loading google news vectors')
    vectors = kv.load_word2vec_format('vectors/google-news.bin', binary=True)
    print('loaded %s vectors' % (len(vectors.vocab)))
    return vectors

load_google_news()
acl1010.load_vectors()