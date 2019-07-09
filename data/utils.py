import re
from nltk.corpus import stopwords


def clean(string):
    stop = set(stopwords.words('english'))  # init stopwords

    string = string.lower()
    string = re.sub(r'<.*?>', ' ', string)  # remove html tags
    string = ' '.join([word for word in string.split() if word not in stop])  # remove stopwords
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)

    string = re.sub(r',', ' , ', string)
    string = re.sub(r'!', ' ! ', string)
    string = re.sub(r'\(', ' \( ', string)
    string = re.sub(r'\)', ' \) ', string)
    string = re.sub(r'\?', ' \? ', string)
    string = re.sub(r' +', ' ', string)

    return string.strip()


def select_features(index, n):
    pass
