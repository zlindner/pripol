import os
import re
import pandas as pd
import numpy as np
from glob import glob
from nltk.corpus import stopwords
from ast import literal_eval

DATA_PRACTICES = [
    'first_party_collection_use', 'third_party_sharing_collection', 'user_choice_control',
    'international_specific_audiences', 'data_security', 'user_access_edit_deletion',
    'policy_change', 'data_retention', 'do_not_track'
]


def load(clean_text=True):
    print('Loading dataset...', end='', flush=True)

    if not os.path.exists('opp-115/opp115.csv'):
        generate_dataset().to_csv('opp-115/opp115.csv', sep=',', index=False)

    data = pd.read_csv('opp-115/opp115.csv', sep=',', header=0)

    if clean_text:
        data['text'] = data['text'].apply(clean)

    print('done!')

    return data


def generate_dataset():
    print('Generating dataset...', end='', flush=True)

    p = load_policies()
    a = load_annotations()

    merged = pd.merge(a, p, on=['policy_id', 'segment_id'], how='outer')
    mode = merged.groupby(['policy_id', 'segment_id']).agg(lambda x: x.value_counts().index[0])
    mode.reset_index(inplace=True)

    print('done!')

    return mode


def load_policies():
    policies = []

    for f in glob('opp-115/sanitized_policies/*.html'):
        with open(f, 'r') as policy:
            text = policy.read()
            segments = text.split('|||')

            p = pd.DataFrame(columns=['policy_id', 'segment_id', 'text'])
            p['segment_id'] = np.arange(len(segments))
            p['policy_id'] = f[27:-5].split('_')[0]
            p['text'] = segments

            policies.append(p)

    p = pd.concat(policies)
    p.reset_index(inplace=True, drop=True)

    return p


def load_annotations():
    annotations = []

    for f in glob('opp-115/annotations/*.csv'):
        a = pd.read_csv(f,
                        sep=',',
                        header=None,
                        names=[
                            'annotation_id', 'batch_id', 'annotator_id', 'policy_id', 'segment_id',
                            'data_practice', 'attributes', 'date', 'url'
                        ])
        a['policy_id'] = f[20:-4].split('_')[0]
        a.drop(['annotation_id', 'batch_id', 'annotator_id', 'date', 'url'], axis=1, inplace=True)
        annotations.append(a)

    a = pd.concat(annotations)
    a.reset_index(inplace=True, drop=True)

    return a


def clean(text):
    stop = set(stopwords.words('english'))

    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop])
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'!', ' ! ', text)
    text = re.sub(r'\(', ' \( ', text)
    text = re.sub(r'\)', ' \) ', text)
    text = re.sub(r'\?', ' \? ', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def practice_counts(data):
    # TODO
    pass


def generate_attribute_distribution(data):
    with open('opp-115/attribute_dist.txt', 'w') as f:
        for dp in data['data_practice'].unique():
            f.write(dp + '\n')

            attributes = get_attribute_counts(data[data['data_practice'] == dp])

            for attr in attributes.keys():
                f.write('\t' + attr + '\n')

                for val, count in attributes[attr].items():
                    f.write('\t\t%s: %s\n' % (val, count))

            f.write('\n')


def get_attribute_counts(data):
    attributes = data['attributes'].to_list()
    counts = {}

    for a in attributes:
        d = literal_eval(a)

        for k, v in d.items():
            if not k in counts:
                counts[k] = {}
            elif not v['value'] in counts[k]:
                counts[k][v['value']] = 1
            else:
                counts[k][v['value']] += 1

    return counts
