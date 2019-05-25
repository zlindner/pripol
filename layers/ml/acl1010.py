from glob import glob
from xml.etree import cElementTree as et
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors as kv
from gensim.models.phrases import Phrases as phrases, Phraser as phraser
import multiprocessing

from collections import defaultdict

# converts the corpus policies from .xml to .txt
def convert():
    for filename in glob('corpus_xml/*.xml'):
        with open(filename, 'r', ) as f_xml:
            policy = et.parse(f_xml).getroot()
            policy_text = ''

            for section in list(policy):
                title = section.find('SUBTITLE').text
                text = section.find('SUBTEXT').text

                if title is not None:
                    policy_text = policy_text + title

                if text is not None:
                    policy_text = policy_text + text

            policy_text = '\n'.join([s for s in policy_text.splitlines() if s])

            with open('corpus_text/' + filename[11:-4] + '.txt', 'w') as f_txt:
                f_txt.write(policy_text)

# loads the corpus into a single string
def load_corpus():
    corpus = ''

    for filename in glob('corpus_text/*.txt'):
        with open(filename, 'r') as f:
            corpus += f.read()

    return corpus

# preprocesses the corpus
def preprocess(corpus):
    corpus = ' '.join(corpus.split()) # remove all whitespace characters
    #TODO other preprocessing

    return corpus

# tokenizes the corpus into a list of lists of tokens
def tokenize(corpus):
    return [word_tokenize(sentence) for sentence in sent_tokenize(corpus)]

def most_frequent(sentences, n):
    word_freq = defaultdict(int)
    
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    
    print(len(word_freq))
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:n])

# trains word2vec vectors using the corpus
def train_vectors(sentences):
    dim_size = 300 # dimension size of vectors
    window_size = 5 # max distance between current and predicted word within a sentence
    min_freq = 1 # ignores all words with a total absolute frequency lower than this
    cores = multiprocessing.cpu_count() # cpu cores

    model = w2v(size=dim_size, window=window_size, min_count=min_freq, workers=cores - 1) # initialize model
    model.build_vocab(sentences) # build vocabulary
    model.train(sentences, total_examples=model.corpus_count, epochs=30) # train model
    model.init_sims(replace=True) # precompute L2-normalized vectors, saves memory, can no longer be trained
    model.save('vectors/acl1010.vec') # write model to disk

# loads word2vec vectors
def load_vectors():
    return kv.load('vectors/acl1010.vec', mmap='r')

#train_vectors(tokenize(preprocess(load_corpus())))

#sentences = tokenize(preprocess(load_corpus()))
#most_frequent(sentences, 20)

#vec = load_vectors()
#print(len(vec.wv.vocab)) # 181

#print(vec.wv.similar_by_word('data'))

#corpus = load_corpus()
#corpus = preprocess(corpus)
#tok = tokenize(corpus)
#print(sentences)

# TODO phrases
#p = phrases(sentences, min_count=30)

#bigram = phraser(p)
#sentences = bigram[sentences]
