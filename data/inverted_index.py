import data.utils as utils


class Database():

    def __init__(self):
        self.db = dict()

    def __repr__(self):
        return str(self.__dict__)

    def get(self, id):
        return self.db.get(id, None)

    def add(self, doc):
        return self.db.update({doc['id']: doc})

    def remove(self, doc):
        return self.db.pop(doc['id'], None)


class Node():

    def __init__(self, id, freq):
        self.id = id
        self.freq = freq

    def __repr__(self):
        return str(self.__dict__)


class InvertedIndex():

    def __init__(self, db):
        self.db = db
        self.index = dict()

    def __repr__(self):
        return str(self.index)

    def index_doc(self, doc):
        text = doc['text']

        text = utils.remove_html(text)
        text = utils.clean(text)

        words = text.split(' ')
        word_freqs = dict()

        for word in words:
            word_freq = word_freqs[word].freq if word in word_freqs else 0
            word_freqs[word] = Node(doc['id'], word_freq + 1)

        update_dict = {
            key: [node] if key not in self.index else self.index[key] + [node] for (key, node) in word_freqs.items()
        }

        self.index.update(update_dict)
        self.db.add(doc)

        return doc

    def lookup(self, query):
        return {word: self.index[word] for word in query.split(' ') if word in self.index}
