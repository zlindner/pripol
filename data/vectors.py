from gensim.models import KeyedVectors as kv


def load_static(name):
    vectors = None

    if name == 'acl':
        print('Loading acl1010 vectors...')
        vectors = kv.load('data/acl1010/acl1010.vec', mmap='r')
    elif name == 'google':
        print('Loading google news vectors...')
        vectors = kv.load_word2vec_format('data/google/google-news.bin', binary=True)
    else:
        print('Invalid static vector name')

    return vectors
