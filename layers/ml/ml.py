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

list_train = []
list_test = []
i = 0

for f in tqdm(glob.glob('opp-115/annotations/*.csv')):
    df = pd.read_csv(f, sep=',', header=None, names=['annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id', 'category', 'attr-vals', 'date', 'url'])
    
    if i < 65:
        list_train.append(df)
    else:
        list_test.append(df)
    
    i = i + 1

df_train = pd.concat(list_train)
df_test = pd.concat(list_test)

print('loaded %s privacy policies containing %s rows for training' % (len(list_train), df_train.shape[0]))
print('loaded %s privacy policies containing %s rows for testing' % (len(list_test), df_test.shape[0]))

# create embedding matrix
print('\n-- creating embedding matrix --\n')

# cnn
print('\n-- training cnn --\n')

#model = Sequential()
#model.add(layers.Embedding(input_dim=$, output_dim=$, input_length=$))
