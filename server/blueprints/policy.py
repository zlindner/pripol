import requests
from flask import Blueprint, request, jsonify
from bs4 import BeautifulSoup

bp = Blueprint('policy', __name__, url_prefix='/policy')


@bp.route('/load', methods=['POST'])
def load_policy():
    url = request.get_json()['url']

    if not url:
        return '', 403

    try:
        html = requests.get(url)
    except Exception:
        return '', 403

    # parse segments from policy html
    segments = parse_html(html)

    return jsonify({'segments': segments}), 200


def parse_html(html):
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

            # unicode to ascii bytes
            p_text = p_text.encode('ascii', 'ignore')

            # ascii bytes => ascii string
            p_text = p_text.decode('ascii')
            segments.append(p_text)

    return segments
