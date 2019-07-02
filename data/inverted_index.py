import data.utils as utils


class InvertedIndex():

    def __init__(self):
        self.index = dict()

    def __repr__(self):
        return str(self.index)

    def index_document(self, document):
        text = document['text']
        terms = text.split(' ')
        term_freqs = dict()

        for term in terms:
            term_freq = term_freqs[term]['freq'] if term in term_freqs else 0
            term_freqs[term] = {'id': document['id'], 'freq': term_freq + 1}

            #Node(document['id'], term_freq + 1)

        update_dict = {
            key: [node] if key not in self.index else self.index[key] + [node] for (key, node) in term_freqs.items()
        }

        self.index.update(update_dict)

    def lookup(self, query):
        return {term: self.index[term] for term in query.split(' ') if term in self.index}
