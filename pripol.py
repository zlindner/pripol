import click
import requests
import string
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# TODO add option to show evaluation results


@click.command()
@click.option(
    '--url',
    help='URL to the web page containing the privacy policy being analyzed')
@click.option('--model', help='')
def cli(url, model):
    segments = scrape_policy(url)


def scrape_policy(url):
    segments = []

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

    for p in paragraphs:
        if p.text.strip != '':
            p_text = p.text
            p_text = p_text.encode('ascii', 'ignore')  # unicode => ascii bytes
            p_text = p_text.decode('ascii')  # ascii bytes => ascii string
            segments.append(p_text)

    print(segments[0])
    segments = clean_policy(segments)
    print(segments[0])

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


if __name__ == '__main__':
    cli()
