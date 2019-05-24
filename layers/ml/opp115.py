from tqdm import tqdm
import glob
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np

pd.options.mode.chained_assignment = None # disable chained assignment warning

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

print('\n-- loading opp-115 --\n')

# load annotations
print('loading annotations')

a_list = [] # list of annotations

for f in tqdm(glob.glob('opp-115/annotations/*.csv')):
	df = pd.read_csv(f, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'category', 'attr-vals', 'date', 'url'])
	df['pretty_print'] = '' # initialize pretty print text to empty string for later
	a_list.append(df)

df_a = pd.concat(a_list) # create single annotation dataframe from list of annotations

# load pretty print
print('loading pretty print')

pp_list = [] # list of pretty print

for f in tqdm(glob.glob('opp-115/pretty_print/*.csv')):
	df = pd.read_csv(f, sep=',', header=None, names=['annotation_id', 'segment_id', 'annotator_id', 'text'])
	pp_list.append(df)

df_pp = pd.concat(pp_list) # create single pretty print dataframe from list of pretty print

# loop through each row of annotation dataframe
print('linking annotations with corresponding pretty printed text')

for i, row in tqdm(df_a.iterrows()):
	match = df_pp.loc[df_pp['annotation_id'] == row['annotation_id']] # locate pretty print row with annotation_id corresponding to current row
	df_a.at[i, 'pretty_print'] = match['text']

# split dataframe into 2 separate for training in testing (training 65%, testing 35%)
size_train = int(df_a.shape[0] / 115 * 65)
df_train, df_test = df_a[:size_train], df_a[size_train:]

df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

print('loaded %s rows for training' %(df_train.shape[0]))
print('loaded %s rows for testing' % (df_test.shape[0]))

# preprocessing 
stop_words = set(stopwords.words('english')) # initialize stopwords
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}']) # add punctuation to stopwords

tokenizer = TreebankWordTokenizer()

# preprocess training data
pretty_print_train = df_train['pretty_print'].tolist() # raw pretty printed text
preprocessed_train = [] # preprocessed pretty printed text

for s in tqdm(pretty_print_train):
	tokens = tokenizer.tokenize(s)
	filtered = [word for word in tokens if word.lower() not in stop_words] # remove stopwords from text
	preprocessed_train.append(" ".join(filtered))

# preprocess testing data
pretty_print_test = df_test['pretty_print'].tolist() # raw pretty printed text
preprocessed_test = [] # preprocessed pretty printed text

for s in tqdm(pretty_print_test):
	if type(s) is not str: continue # required due to bug in annotation software, see manual.txt: Errant Span Indexes

	tokens = tokenizer.tokenize(s)
	filtered = [word for word in tokens if word.lower() not in stop_words] # remove stopwords from text
	preprocessed_test.append(" ".join(filtered))

# tokenize data
tokenizer = Tokenizer(num_words=100000, lower=True, char_level=False)
tokenizer.fit_on_texts(preprocessed_train + preprocessed_test)

seq_train = tokenizer.texts_to_sequences(preprocessed_train)
seq_test = tokenizer.texts_to_sequences(preprocessed_test)

word_index = tokenizer.word_index
print('dictionary size: ', len(word_index))

df_train['len'] = df_train['pretty_print'].apply(lambda words: len(words.split(' ')))
max_seq_len = np.round(df_train['len'].mean() + df_train['len'].std()).astype(int)

seq_train = sequence.pad_sequences(seq_train, maxlen=max_seq_len)
seq_test = sequence.pad_sequences(seq_test, maxlen=max_seq_len)

# create embedding matrix
