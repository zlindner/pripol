import re
from nltk.corpus import stopwords

# removes html tags from string
def remove_html(string):
	string = re.sub(r'<.*?>', '', string)

	return string

# removes stopwords, contractions, and whitespaces from string
def clean(string):
	stop = set(stopwords.words('english')) # init stopwords

	string = string.lower()
	string = ' '.join([word for word in string.split() if word not in stop])
	string = re.sub(r'[^A-Za-z0-9(),!?|\'\`]', ' ', string)
	string = re.sub(r'\'s', ' \'s', string)
	string = re.sub(r'\'ve', ' \'ve', string)
	string = re.sub(r'n\'t', ' n\'t', string)
	string = re.sub(r'\'re', ' \'re', string)
	string = re.sub(r'\'d', ' \'d', string)
	string = re.sub(r'\'ll', ' \'ll', string)
	string = re.sub(r',', ' , ', string)
	string = re.sub(r'!', ' ! ', string)
	string = re.sub(r'\(', ' \( ', string)
	string = re.sub(r'\)', ' \) ', string)
	string = re.sub(r'\?', ' \? ', string)
	string = re.sub(r'\s{2,}', ' ', string)

	return string.strip()