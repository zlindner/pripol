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


def load_non_static():
    vectors = kv.load('data/acl1010/acl1010.vec', mmap='r')
    vectors.intersect_word2vec_format('data/google/google-news.bin', lockf=1.0, binary=True)

    return vectors


def demo():
    print('\n### Vector Demo ###\n')

    acl = load_static('acl')
    google = load_static('google')

    while True:
        word = input('Enter a word: ')

        try:
            similar = acl.similar_by_word(word)
            print('ACL1010 similar words\n%s\n' % similar)
        except KeyError:
            print('Word %s is not in the acl1010 vocabulary', word)

        try:
            similar = google.similar_by_word(word)
            print('Google-news similar words\n%s\n' % similar)
        except KeyError:
            print('Word %s is not in the google-news vocabulary', word)
