from tqdm import tqdm
import glob
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np

#pd.options.mode.chained_assignment = None # disable chained assignment warning

# tokenize data
#tokenizer = Tokenizer(num_words=100000, lower=True, char_level=False)
#tokenizer.fit_on_texts(preprocessed_train + preprocessed_test)

#seq_train = tokenizer.texts_to_sequences(preprocessed_train)
#seq_test = tokenizer.texts_to_sequences(preprocessed_test)

#word_index = tokenizer.word_index
#print('dictionary size: ', len(word_index))

#df_train['len'] = df_train['pretty_print'].apply(lambda words: len(words.split(' ')))
#max_seq_len = np.round(df_train['len'].mean() + df_train['len'].std()).astype(int)

#seq_train = sequence.pad_sequences(seq_train, maxlen=max_seq_len)
#seq_test = sequence.pad_sequences(seq_test, maxlen=max_seq_len)

# create embedding matrix
