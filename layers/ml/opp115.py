from glob import glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#import cnn
#model = cnn.model

# the 10 data practice categories from https://usableprivacy.org/static/files/swilson_acl_2016.pdf
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

# loads annotations
def load_annotations():
	print('\n# loading annotations')

	annotations = [] # list of annotation dataframes

	for filename in glob('opp-115/annotations/*.csv'):
		df = pd.read_csv(filename, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'category', 'attr_vals', 'date', 'url'])
		df['pretty_print'] = ''	
		df['filename'] = ''
		annotations.append(df)

	df = pd.concat(annotations) # create single dataframe from list
	df.reset_index(inplace=True)

	print('loaded %s annotations for %s privacy policies' % (df.shape[0], len(annotations)))

	return df

# loads pretty prints
def load_pretty_prints():
	print('\n# loading pretty prints')

	pretty_prints = [] # list of pretty prints

	for filename in glob('opp-115/pretty_print/*.csv'):
		df = pd.read_csv(filename, sep=',', header=None, names=['annotation_id', 'segment_id', 'annotator_id', 'pretty_print'])
		df['filename'] = filename[21:] # track filename
		pretty_prints.append(df)

	df = pd.concat(pretty_prints)
	df.reset_index(inplace=True)

	print('loaded %s pretty prints for %s privacy policies' % (df.shape[0], len(pretty_prints)))

	return df

# links annotations and their corresponding pretty prints by annotation id
def link_dataframes(annotations, pretty_prints):
	print('\n# linking annotations with pretty prints')

	for i, row in annotations.iterrows():
		link = pretty_prints.loc[pretty_prints['annotation_id'] == row['annotation_id']] 
		annotations.at[i, 'pretty_print'] = link['pretty_print']
		annotations.at[i, 'filename'] = link['filename']

	print('updated %s rows' % (annotations.shape[0]))

# generates various statistics pertaining to the opp115
def generate_statistics(annotations):
	print('\n# generating statistics')

	category_freqs = annotations['category'].value_counts()

	print(category_freqs)

# TODO
def encode_sequences(X_train, X_val):
	print('encoding text sequences...')

	tokenizer = Tokenizer()
	print(X_train['pretty_print'].values.tolist()[0])
	#tokenizer.fit_on_texts(X_train['pretty_print'].tolist())

# splits the dataframe into training and testing subsets 
def split_train_test(df):
	print('\n# splitting into training, testing, and validation subsets')

	y = df.pop('category')
	X = df
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0766, random_state=42)

	return X_train, X_test, y_train, y_test

# partitions X into training and validation subsets n times
def split_stratified_kfold(X, y, n):
	skf = StratifiedKFold(n_splits=n, random_state=42)
	
	for train_index, val_index in skf.split(X, y):
		X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(val_index)]
		y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(val_index)]

		encode_sequences(X_train, X_val)

		#model.fit(X_train, y_train, epochs=10)

annotations = load_annotations()
pretty_prints = load_pretty_prints()
link_dataframes(annotations, pretty_prints)

generate_statistics(annotations)

X_train, X_test, y_train, y_test = split_train_test(annotations)
split_stratified_kfold(X_train, y_train, 10)
