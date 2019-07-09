import pandas as pd
import data.utils as utils

from data.inverted_index import InvertedIndex


class Corpus():

    DATA_PRACTICES = [
        'first_party_collection_use',
        'third_party_sharing_collection',
        'introductory_generic',
        'user_choice_control',
        'international_specific_audiences',
        'data_security',
        'privacy_contact_information',
        'user_access_edit_deletion',
        'practice_not_covered',
        'policy_change',
        'data_retention',
        'do_not_track'
    ]

    def __init__(self):
        self.opp115 = None
        self.iindex = InvertedIndex()
        self.metadata = {}  # map of relative document id -> policy id

    def load(self):
        '''Loads the opp115 corpus'''

        print('Loading corpus...')

        self.opp115 = pd.read_csv('data/corpus/opp115.csv', sep=',', header=0)

        # self.opp115 = self.opp115[(self.opp115['data_practice'] != 'Introductory/Generic') &
        #                          (self.opp115['data_practice'] != 'Privacy contact information') &
        #                          (self.opp115['data_practice'] != 'Practice not covered') &
        #                          (self.opp115['data_practice'] != 'Other')]
        #self.opp115['segment'] = self.opp115['segment'].apply(utils.clean)
        # self.opp115.reset_index(inplace=True)

        self.index()

        return self.opp115

    def index(self):
        '''Creates an inverted index containing document frequencies for each word in the corpus'''

        print('Indexing documents...')

        policies = list(self.opp115.groupby(['policy_id']))
        id = 0

        for policy in policies:
            document = {
                'text': ' '.join(policy[1]['segment'].values),
                'id': id
            }

            self.metadata[id] = policy[0]
            self.iindex.index_document(document)

            id += 1

        # calculate term frequencies
        for term in self.iindex.index:
            term_freq = sum([node['freq'] for node in self.iindex.index[term]])
            self.iindex.index[term].append({'id': -1, 'freq': term_freq})

    def generate_statistics(self):
        '''Generates statistics for the opp115 corpus. load() must be called before generating statistics'''

        if self.opp115 is None:
            print('OPP115 hasn\'t been loaded yet, can\'t generate statistics')
            return

        print('Generating statistics...')

        total_segments = self.opp115.shape[0]
        total_words = self.opp115['segment'].apply(lambda x: len(x.split(' '))).sum()
        avg_words = self.opp115['segment'].apply(lambda x: len(x.split(' '))).mean()
        unique_words = len(self.iindex.index)
        distribution = self.opp115[Corpus.DATA_PRACTICES].apply(pd.Series.value_counts).values[1]
        distribution = dict(zip(Corpus.DATA_PRACTICES, distribution))

        self.statistics = {
            'total_segments': total_segments,
            'total_words': total_words,
            'avg_words': avg_words,
            'unique_words': unique_words,
            'distribution': distribution
        }

        return self.statistics
