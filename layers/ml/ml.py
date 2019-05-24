from tqdm import tqdm
import glob
import pandas as pd
from keras.models import Sequential
from keras import layers

# load embeddings
print('\n-- loading word embeddings --\n')

embeddings = {}

with open('vectors/wiki-simple.vec', 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    for line in tqdm(f):
        tokens = line.rstrip().split(' ')
        embeddings[tokens[0]] = map(float, tokens[1:])
        
print('loaded %s word vectors' % (len(embeddings)))

# load opp-115 (65 policies for training, 50 for testing) TODO use consolidated .csv's???
print('\n-- loading opp-115 --\n')

# create embedding matrix
print('\n-- creating embedding matrix --\n')

# cnn
print('\n-- training cnn --\n')

#model = Sequential()
#model.add(layers.Embedding(input_dim=$, output_dim=$, input_length=$))
