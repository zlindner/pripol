import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

	# initialize opp115
	def __init__(self):
		self.corpus = self.load()

	# loads opp115
	def load(self):
		print('\n# loading opp115')

		annotations = self.load_annotations()
		pretty_prints = self.load_pretty_print()

		opp115 = self.link(annotations, pretty_prints)

		return opp115	

	# loads annotations for each privacy policy
	def load_annotations(self):
		print('loading annotations...')

		annotations = []

		for filename in glob('data/opp115/annotations/*.csv'):
			df = pd.read_csv(filename, sep=',', header=None, names=['id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'data_practice', 'attr_vals', 'date', 'url'])
			annotations.append(df)

		df = pd.concat(annotations)
		df.drop(['batch_id', 'annotator_id', 'policy_id', 'segment_id', 'date', 'url'], axis=1, inplace=True)

		print('loaded %s annotations for %s privacy policies' % (df.shape[0], len(annotations)))

		return df

	# loads pretty_prints for each privacy policy
	def load_pretty_print(self):
		print('loading pretty prints...')

		pretty_prints = []

		for filename in glob('data/opp115/pretty_print/*.csv'):
			df = pd.read_csv(filename, sep=',', header=None, names=['id', 'segment_id', 'annotator_id', 'text'])
			df['filename'] = filename
			pretty_prints.append(df)
			
		df = pd.concat(pretty_prints)
		df.drop(['segment_id', 'annotator_id'], axis=1, inplace=True)

		print('loaded %s pretty prints for %s privacy policies' % (df.shape[0], len(pretty_prints)))

		return df
		
	# links annotations with pretty_prints by id
	def link(self, annotations, pretty_prints):
		print('merging annotations with pretty prints...')
		
		df = pd.merge(annotations, pretty_prints, on='id')
		print('linked %s rows' % (df.shape[0]))
		
		return df

	# partitions the opp115 into training and testing subsets TODO stratified kfold on training set
	def partition(self, opp115):
		train, test = train_test_split(opp115, test_size=0.0766, random_state=42, stratify=opp115[['data_practice']])

		return train, test

	# builds text sequences for the training and testing sets
	def build_sequences(self):
		print('\n# builiding text sequences')

		train, test = self.partition(self.corpus)
		train = train['text'].tolist()
		test = test['text'].tolist()

		max_sentence_length = max(max([len(sentence.split()) for sentence in train]), max([len(sentence.split()) for sentence in test]))

		tokenizer = Tokenizer(oov_token=True)
		tokenizer.fit_on_texts(train)

		seq_train = tokenizer.texts_to_sequences(train)
		seq_test = tokenizer.texts_to_sequences(test)

		X_train = pad_sequences(seq_train, maxlen=max_sentence_length, padding='post')
		X_test = pad_sequences(seq_test, maxlen=max_sentence_length, padding='post')

		vocab_size = len(tokenizer.word_index) + 1

		return X_train, X_test, vocab_size
