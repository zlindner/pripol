import re
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

# TODO load unconsolidated annotations, compare predicition accuracy?
# TODO load pretty print?
class OPP115:

	DATA_PRACTICES = [
		'First Party Collection/Use',
		'Third Party Sharing/Collection',
		'User Choice/Control',
		'User Access, Edit, & Deletion',
		'Data Retention',
		'Data Security',
		'Policy Change',
		'Do Not Track',
		'International & Specific Audiences',
		'Other'
	]

	def __init__(self):
		self.consolidated = self.load_consolidated(0.5)
		self.X_train, self.X_test, self.y_train, self.y_test = self.build_sequences()

	# load consolidated annotations with overlap similarities of 0.5, 0.75, or 1.0
	def load_consolidated(self, threshold):
		if threshold != 0.5 and threshold != 0.75 and threshold != 1.0:
			print('error loading consolidations with threshold of %s' % threshold)
			return

		print('\n# loading consolidations with threshold %s...' % threshold)

		consolidated = []

		for filename in glob('data/opp115/consolidation/threshold-' + str(threshold) + '-overlap-similarity/*.csv'):
			consol = pd.read_csv(filename, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attributes', 'date', 'url'])
			consol.drop(['policy_id', 'date', 'url'], axis=1, inplace=True) # remove extra columns
			
			filename_policy = filename[60:-4] if threshold == 0.75 else filename[59:-4] # threshold of 0.75 has extra character in filename
			policy = self.load_policy(filename_policy) # load policy corresponding to annotations
			
			merged = pd.merge(consol, policy, on='segment_id') # merge each annotation with its corresponding segment in the policy text
			
			consolidated.append(merged)

		df = pd.concat(consolidated)
		df.reset_index(inplace=True)
		
		print('loaded %s annotations for %s privacy policies' % (df.shape[0], len(consolidated)))

		return df

	# loads the sanitized policy for the given filename, splitting it into clean segments
	def load_policy(self, filename):
		policies = []
		
		with open('data/opp115/sanitized_policies/' + filename + '.html', 'r') as f:
			html = f.read() # sanitized policy
			
			policy = re.sub(r'<.*?>', '', html) # remove html tags
			policy = self.clean(policy)
					
			segments = policy.split('|||')

			df = pd.DataFrame(columns=['segment_id', 'segment'])
			df['segment_id'] = np.arange(0, len(segments))
			df['segment'] = segments

			return df
	
	# cleans the passed text
	def clean(self, text):
		text = ' '.join(text.split()) # remove extra whitespaces
		text = re.sub(r'[\(\)\[\]\{\}\;\:\`\"\“\”]', '', text) # remove punctuation
		text.replace(r'http\S+', '')
		text.replace(r'http', '')
		text.replace(r'@\S+', '')
		text.replace(r'@', 'at')
		text = text.lower()

		return text

	# partitions the annotations into training and testing subsets TODO stratified kfold on training set
	def partition(self, annotations):
		X = annotations['segment'].values
		y = annotations['data_practice'].values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0766, random_state=42, stratify=annotations[['data_practice']])

		return X_train, X_test, y_train, y_test

	#
	def encode_labels(self, labels):
		enc = LabelEncoder()
		y = enc.fit_transform(labels)

		#self.encoder = OneHotEncoder(sparse=False)
		#y = labels.reshape((-1, 1))
		#y = self.encoder.fit_transform(y)

		y = to_categorical(y)

		return y

	#
	def build_sequences(self):
		print('\n# building text sequences')

		# 
		X_train, X_test, y_train, y_test = self.partition(self.consolidated)
		self.max_seq_len = max(max([len(sentence.split()) for sentence in X_train]), max([len(sentence.split()) for sentence in X_test]))

		#
		y_train = self.encode_labels(y_train)
		y_test = self.encode_labels(y_test)

		# 
		self.tokenizer = Tokenizer()
		self.tokenizer.fit_on_texts(X_train)

		self.vocab = self.tokenizer.word_index
		self.vocab_size = len(self.tokenizer.word_index) + 1

		X_train = self.tokenizer.texts_to_sequences(X_train)
		X_test = self.tokenizer.texts_to_sequences(X_test)

		X_train = pad_sequences(X_train, maxlen=self.max_seq_len, padding='post')
		X_test = pad_sequences(X_test, maxlen=self.max_seq_len, padding='post')

		return X_train, X_test, y_train, y_test