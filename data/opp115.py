import re
import pandas as pd
import numpy as np
import data.utils as utils

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

class OPP115:

	def __init__(self):
		print('\n# OPP-115\n')

		self.stats = {
			'documents': -1,
			'classes': 10,
			'total_segments': -1,
			'total_words': -1,
			'cleaned_words': -1,
			'data_practices': [],
			'consolidated_data_practices': []
		}

		self.annotations = self.load_annotations()
		self.stats['data_practices'] = self.annotations['data_practice'].value_counts()

		self.policies = self.load_policies()
		self.linked = self.link()

		self.consolidated = self.consolidate()
		self.stats['total_words'] = self.consolidated['segment'].apply(lambda x: len(x.split(' '))).sum()

		self.consolidated['segment'] = self.consolidated['segment'].apply(utils.remove_html)
		self.consolidated['segment'] = self.consolidated['segment'].apply(utils.clean)
		self.stats['cleaned_words'] = self.consolidated['segment'].apply(lambda x: len(x.split(' '))).sum()

		self.encoded = self.encode()

		self.data_practices = ['other', 'policy_change', 'first_party_collection_use', 'third_party_sharing_collection', 'do_not_track', 'user_choice_control', 'international_specific_audiences', 'data_security', 'data_retention', 'user_access_edit_deletion']
		
		self.stats['consolidated_data_practices'] = self.consolidated['data_practice'].value_counts()
	#
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

	#
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

	#
	def link(self):
		print('linking annotations with segments...')

		return pd.merge(self.annotations, self.policies, on=['policy_id', 'segment_id'], how='outer')

	#
	def consolidate(self):
		print('consolidating annotations...')

		consolidated = self.linked.groupby(['policy_id', 'segment_id']).agg(lambda x: x.value_counts().index[0])
		consolidated.reset_index(inplace=True)
		return consolidated

	#
	def encode(self):
		data_practices = {
			'Other': 'other',
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
		print('Number of classes: %s' % self.stats['classes'])
		print('Number of segments: %s' % self.stats['total_segments'])
		print('Number of words: %s' % self.stats['total_words'])
		print('Number of words after cleaning: %s' % self.stats['cleaned_words'])
		print('Data practice distribution:\n%s\n' % self.stats['data_practices'])
		print('Consolidated data practice distribution:\n%s\n' % self.stats['consolidated_data_practices'])

#y_train = np.argmax(opp115.y_train, axis=1)
#y_test = np.argmax(opp115.y_test, axis=1)

#nb.fit(opp115.x_train, opp115.y_train)

#print(opp115.y_test.value_counts())
#y_pred = nb.predict(opp115.x_test)

#data_practices = ['other', 'policy_change', 'first_party_collection_use', 'third_party_sharing_collection', 'do_not_track', 'user_choice_control', 'international_specific_audiences', 'data_security', 'data_retention', 'user_access_edit_deletion']

#print('accuracy %s' % accuracy_score(y_pred, opp115.y_test))
#print(classification_report(opp115.y_test, y_pred))
#print(confusion_matrix(opp115.y_test, y_pred))