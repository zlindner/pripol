import click
import requests
import string
import re
import tensorflow as tf
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

import opp115
from cnn import CNN

# TODO create data.py for handling loading / creating data

@click.command()
@click.option('--url', '-u', help='The web page\'s privacy policy url.')
@click.option('--name', '-n', help='The model to use for the coverage analysis. Models currently supported: lstm (Long Short-Term Memory), cnn (Convolutional Neural Network).')
@click.option('--train', '-t', is_flag=True, help='Retrain the model.')
def cli(url, name, train):
    segments = scrape_policy(url)
    clean = clean_policy(segments)
    
    if train:
        model = train_model(name)
    else:
        model = load_model(name)

    embeddings = load_embeddings()
    x, vocab = init_sequences(clean)
    matrix = init_matrix(vocab, embeddings)

    y_predict = model.predict(x)
    y_classes = np.argmax(y_predict, axis=1)

    for i, segment in enumerate(segments):
        print('%s: %s' % (segment, opp115.DATA_PRACTICES[y_classes[i]]))
    
def scrape_policy(url):
    try:
        html = requests.get(url)
    except Exception:
        print('Invalid privacy policy url supplied.')
        exit()

    soup = BeautifulSoup(html.text, features='html.parser')

    [head.decompose() for head in soup('head')]
    [header.decompose() for header in soup('header')]
    [footer.decompose() for footer in soup('footer')]
    [script.decompose() for script in soup('script')]
    [nav.decompose() for nav in soup('nav')]

    paragraphs = soup.find_all('p')
    segments = []

    for p in paragraphs:
        if p.text.strip != '':
            p_text = p.text
            p_text = p_text.encode('ascii', 'ignore')  # unicode => ascii bytes
            p_text = p_text.decode('ascii')  # ascii bytes => ascii string
            segments.append(p_text)

    return segments

def clean_policy(segments):
    clean = []
    stop_words = set(stopwords.words('english'))

    for i in range(len(segments)):
        clean.append(segments[i].lower())

        # remove punctuation
        clean[i] = clean[i].translate(str.maketrans('', '', string.punctuation))

        # remove stop words
        clean[i] = ' '.join([word for word in clean[i].split() if word not in stop_words])

        # remove extra whitespace
        clean[i] = re.sub(r' +', ' ', clean[i])
        clean[i] = clean[i].strip()

    return clean

def load_embeddings():
    print('Loading word embeddings...', end='', flush=True)
    embeddings = KeyedVectors.load('acl-1010/acl1010.vec', mmap='r')
    print('done!')
    
    return embeddings

def init_sequences(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    vocab = tokenizer.word_index

    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=100, padding='post')
    
    return x, vocab

def init_matrix(vocab, vec):
    vocab_size = len(vocab) + 1
    matrix = np.zeros((vocab_size, 300))

    for word, i in vocab.items():
        if i >= vocab_size:
            continue

        if word in vec.wv.vocab:
            vector = vec[word]

            if vector is not None and len(vector) > 0:
                matrix[i] = vector

    return matrix

def load_model(name):
    try:
        path = './models/' + name + '.h5'
        model = tf.keras.models.load_model(path)
    except OSError:
        print('Invalid model name supplied.')
        exit()

    return model

def train_model(name):
    if name == 'cnn':
        model = CNN()
    elif name == 'lstm':
        pass
    else:
        print('Invalid model name supplied.')
        exit()

    model.create()
    model.train()
    model.save()

    return model

if __name__ == '__main__':
    cli()
