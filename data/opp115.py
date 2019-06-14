import re
import pandas as pd
import numpy as np
import data.utils as utils

from glob import glob
from ast import literal_eval
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class OPP115:

	def __init__(self):
		print('\n# OPP-115\n')

		self.stats = {
			'documents': -1,
			'total_segments': -1,

			'words_total': -1,
			'words_clean': -1,
			'word_stem': -1,

			'data_practices': [],
			'data_practices_expanded': [],
			'data_practices_consolidated': []
		}

		self.annotations = self.load_annotations()
		self.stats['data_practices'] = self.annotations['data_practice'].value_counts()

		# expand 'Other' data practice into sub-datapractices
		self.annotations['data_practice'] = self.annotations.apply(self.expand_other, axis=1)
		self.stats['data_practices_expanded'] = self.annotations['data_practice'].value_counts()
		
		# load and link policies
		self.policies = self.load_policies()
		self.linked = self.link()

		# consolidate multiple annotations for a segment into one
		self.consolidated = self.consolidate()
		self.stats['words_total'] = self.consolidated['segment'].apply(lambda x: len(x.split(' '))).sum()
		self.stats['data_practices_consolidated'] = self.consolidated['data_practice'].value_counts()

		# clean text of segments
		self.consolidated['segment'] = self.consolidated['segment'].apply(utils.remove_html)
		self.consolidated['segment'] = self.consolidated['segment'].apply(utils.clean)
		self.stats['words_clean'] = self.consolidated['segment'].apply(lambda x: len(x.split(' '))).sum()

		# apply stemming to each word in a segment TODO not sure why this results in more words???
		stemmer = PorterStemmer()
		self.consolidated['segment'] = self.consolidated['segment'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)])) 
		self.stats['words_stem'] = self.consolidated['segment'].apply(lambda x: len(x.split(' '))).sum()

		# one hot encode data practices
		self.encoded = self.encode()

	# loads annotations
	def load_annotations(self):
		print('loading annotations...')

		annotations = []

		for filename in glob('data/opp115/annotations/*.csv'):
			df = pd.read_csv(filename, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attributes', 'date', 'url'])
			df.drop(['annotation_id', 'batch_id', 'annotator_id', 'date', 'url'], axis=1, inplace=True)
			df['policy_id'] = filename[24:-4].split('_')[0]
			annotations.append(df)

		df = pd.concat(annotations)
		df.reset_index(inplace=True, drop=True)

		print('loaded %s annotations' % df.shape[0])

		self.stats['documents'] = len(annotations)

		return df

	# expands 'Other' data practice into subsets
	def expand_other(self, x):
		if x['data_practice'] != 'Other':
			return x['data_practice']

		attrs = literal_eval(x['attributes'])
		
		return attrs['Other Type']['value']

	# loads sanitized privacy policies
	def load_policies(self):
		print('loading policies...')

		policies = []

		for filename in glob('data/opp115/sanitized_policies/*.html'):
			with open(filename, 'r') as f:
				policy = f.read()
				segments = policy.split('|||')

				df = pd.DataFrame(columns=['policy_id', 'segment_id', 'segment'])
				df['segment_id'] = np.arange(0, len(segments))
				df['segment'] = segments
				df['policy_id'] = filename[31:-5].split('_')[0]

				policies.append(df)

		df = pd.concat(policies)
		df.reset_index(inplace=True, drop=True)

		self.stats['total_segments'] = df.shape[0]

		print('loaded %s segments' % df.shape[0])

		return df		

	# links annotations with segments based on policy_id and segment_id
	def link(self):
		print('linking annotations with segments...')
		return pd.merge(self.annotations, self.policies, on=['policy_id', 'segment_id'], how='outer')

	# consolidates annotations by taking the mode data practice for each segment
	def consolidate(self):
		print('consolidating annotations...')

		consolidated = self.linked.groupby(['policy_id', 'segment_id']).agg(lambda x: x.value_counts().index[0])
		consolidated.reset_index(inplace=True)
		return consolidated

	# encodes data practice into one hot format
	def encode(self):
		data_practices = {
			'Other': 'other',
			'Introductory/Generic': 'introductory_generic',
			'Privacy contact information': 'privacy_contact_information',
			'Practice not covered': 'practice_not_covered',
			'Policy Change': 'policy_change',
			'First Party Collection/Use': 'first_party_collection_use',
			'Third Party Sharing/Collection': 'third_party_sharing_collection',
			'Do Not Track': 'do_not_track',
			'User Choice/Control': 'user_choice_control',
			'International and Specific Audiences': 'international_specific_audiences',
			'Data Security': 'data_security',
			'Data Retention': 'data_retention',
			'User Access, Edit and Deletion': 'user_access_edit_deletion'
		}

		encoded = pd.DataFrame({'policy_id': self.consolidated['policy_id'], 'segment_id': self.consolidated['segment_id']})

		for data_practice in data_practices:
			one_hot = lambda x: 1 if x.startswith(data_practice) else 0
			encoded[data_practices[data_practice]] = self.consolidated['data_practice'].apply(one_hot)

		return encoded

	# displays various statistics about the opp115 dataset
	def display_statistics(self):
		print('\n# opp115 statistics')
		
		print('Total documents: %s' % self.stats['documents'])
		print('Number of segments: %s' % self.stats['total_segments'])
		print('Number of words: %s' % self.stats['words_total'])
		print('Number of words after cleaning: %s' % self.stats['words_clean'])
		print('Number of words after stemming: %s' % self.stats['words_stem'])
		print('Data practice distribution:\n%s\n' % self.stats['data_practices'])
		print('Data practice distribution after expanding Other:\n%s\n' % self.stats['data_practices_expanded'])
		print('Consolidated data practice distribution:\n%s\n' % self.stats['data_practices_consolidated'])
			