import click
import requests
import string
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from cnn import CNN

# TODO add --show_results option
# TODO add --retrain option


@click.command()
@click.option('--url', '-u', help='URL to the web page containing the privacy policy being analyzed.')
@click.option('--model', '-m', help='The model to use for the coverage analysis. Models currently supported: lstm (Long Short-Term Memory), cnn (Convolutional Neural Network).')
@click.option('--train', '-t', is_flag=True, help="Train / retrain the model.")
def cli(url, model, train):
    segments = scrape_policy(url)
    
    if train:
        train_model(model)
    else:
        try:
            load_model(model)
        except OSError:
            print('Invalid model name supplied.')


def scrape_policy(url):
    try:
        html = requests.get(url)
    except Exception:
        print('Invalid privacy policy url supplied.')

    soup = BeautifulSoup(html.text, features='html.parser')

    [head.decompose() for head in soup('head')]
    [header.decompose() for footer in soup('header')]
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

    segments = clean_policy(segments)

    return segments


def clean_policy(segments):
    stop_words = set(stopwords.words('english'))

    for i in range(len(segments)):
        segments[i] = segments[i].lower()

        # remove punctuation
        segments[i] = segments[i].translate(str.maketrans('', '', string.punctuation))

        # remove stop words
        segments[i] = ' '.join([word for word in segments[i].split() if word not in stop_words])

        # remove extra whitespace
        segments[i] = re.sub(r' +', ' ', segments[i])
        segments[i] = segments[i].strip()

    return segments

def train_model(name):
    if name == 'cnn':
        cnn = CNN()
    elif name == 'lstm':
        pass
    else:
        print('Invalid model name supplied.')
        exit()

if __name__ == '__main__':
    cli()
