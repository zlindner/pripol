import pandas as pd
import re
from glob import glob
from xml.etree import cElementTree as et
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec as w2v
from gensim.models import KeyedVectors as kv
from collections import defaultdict

class ACL1010:
	
	# converts the corpus policies from .xml to .csv, split into sections
	def convert(self):
		print('\n# converting acl1010 from xml to csv')

		for filename in glob('acl1010/xml/*.xml'):
			sections = []

			with open(filename, 'r', ) as f:
				policy = et.parse(f).getroot()

				for section in list(policy):
					subtitle = section.find('SUBTITLE').text
					subtext = section.find('SUBTEXT').text

					subtitle = clean(subtitle) if subtitle is not None else '_NA_'
					subtext = clean(subtext) if subtext is not None else '_NA_'

					sections.append({
						'subtitle': subtitle,
						'text': subtext,
						'filename': filename[12:-4] 
					})
				
				df = pd.DataFrame(sections)		
				df.to_csv('data/acl1010/csv/' + filename[12:-4] + '.csv', sep=',', encoding='utf-8', index=False)

	# preprocesses xml text before writing to csv
	def clean(self, text):
		text = ' '.join(text.split()) # remove all whitespace characters
		text = text.lower() # lowercase
		text = re.sub(r'[\(\)\[\]\{\}\;\:\`\"\“\”]', '', text) # remove punctuation
		text.replace(r'http\S+', '')
		text.replace(r'http', '')
		text.replace(r'@\S+', '')
		text.replace(r'@', 'at')
		
		return text
				
	# loads acl1010 privacy policies
	def load(self):
		print('\n# loading acl1010')

		policies = []

		for filename in glob('data/acl1010/csv/*.csv'):
			df = pd.read_csv(filename, sep=',', header=0, index_col=0)
			policies.append(df)

		df = pd.concat(policies)
		df.reset_index(inplace=True)

		print('loaded %s sections from %s privacy policies' % (df.shape[0], len(policies)))

		return df

	# tokenizes the acl1010 and removes stopwords TODO pick random sample -> check accuracy
	def preprocess(self, acl1010):
		print('\n# preprocessing acl1010')

		stop = set(stopwords.words('english')) # init stopwords
		stop.update(['_NA_']) # TODO possibly remove .,!? as section boundaries are already established

		print('tokenizing...')
		subtitles = acl1010['subtitle'].apply(lambda subtitle: [word_tokenize(sentence) for sentence in sent_tokenize(subtitle)])
		texts = acl1010['text'].apply(lambda text: [word_tokenize(sentence) for sentence in sent_tokenize(text)])
		sections = subtitles + texts # concatenate subtitle and text

		print('removing stopwords...')
		acl1010['section'] = sections.apply(lambda section: [[word for word in sentence if word not in stop] for sentence in section])
		acl1010['section'] = acl1010['section'].apply(lambda section: [sentence for sentence in section if sentence]) # remove empty lists

		return acl1010

	# trains word vectors from the acl1010
	def train(self, acl1010):
		print('\n# training word vectors')
		sentences = [sentence for section in acl1010['section'] for sentence in section]

		print('initializing model...')
		model = w2v(size=300, window=5, min_count=1, workers=4)
		model.build_vocab(sentences)

		print('training...')
		model.train(sentences, total_examples=model.corpus_count, epochs=30)
		model.init_sims(replace=True) # precompute L2-normalized vectors, saves memory, can no longer be trained
		model.save('data/acl1010/vec/acl1010.vec') # write model to disk

	def load_vectors(self):
		print('\n# loading word vectors')

		vectors = kv.load('data/acl1010/acl1010.vec', mmap='r')
		print('loaded %s vectors' % (len(vectors.wv.vocab)))
		
		return vectors

	# displays the n most frequent words in the acl1010
	def most_frequent(self, acl1010, n):
		freq = defaultdict(int)
		
		for section in acl1010['section']:
			for sentence in section:
				for word in sentence:
					freq[word] += 1
		
		print(sorted(freq, key=freq.get, reverse=True)[:n])