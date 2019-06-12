import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, cross_validate
from sklearn.exceptions import UndefinedMetricWarning

class Model():

	def __init__(self, opp115):
		print('\n# %s' % self.get_name())

		self.opp115 = opp115

		# initialize model
		self.model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('classifier', self.classifier)])

		# initialize metrics
		self.metrics = {
			'precision': make_scorer(precision_score),
			'recall': make_scorer(recall_score),
			'f1': make_scorer(f1_score),
			'f1_micro': make_scorer(f1_score),
			'f1_macro': make_scorer(f1_score),
			'tp': make_scorer(self.tp),
			'tn': make_scorer(self.tn),
			'fp': make_scorer(self.fp),
			'fn': make_scorer(self.fn)
		}

		# disable UndefinedMetricWarning
		warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

	# returns the name of the model
	def get_name(self):
		raise NotImplementedError

	# perform 10-fold cross validation
	def kfold(self):
		for data_practice in self.opp115.data_practices:
			print('\n--- %s ---\n' % data_practice.upper())

			target = self.opp115.encoded[['policy_id', 'segment_id', data_practice]]
			target = self.opp115.consolidated.merge(target, on=['policy_id', 'segment_id']).drop_duplicates()
			x = target['segment']
			y = target[data_practice]

			kf = KFold(n_splits=10, random_state=42)

			results = cross_validate(estimator=self.model, X=x, y=y, cv=kf, scoring=self.metrics)

			self.display_metrics(results)

	# displays various metrics for kfold cross validation
	def display_metrics(self, results):
		# display headers
		for i, d in enumerate([list(self.metrics.keys())]):
			line = '|'.join(str(x).ljust(12) for x in d)
			print('\t|' + line)
			if i == 0:
				print('-' * len(line))

		# display metrics for each fold 
		for i in range(0, 10):
			fold_results = [results['test_' + metric][i] for metric in self.metrics.keys()]
			line = '|'.join(str(round(x, 4)).ljust(12) for x in fold_results)
			print('fold %s\t|%s' % ((i + 1), line))

		# display mean for each metric not including confusion matrix
		mean = [np.mean(results['test_precision']), np.mean(results['test_recall']), np.mean(results['test_f1']), np.mean(results['test_f1_micro']), np.mean(results['test_f1_macro'])]
		line = '|'.join(str(round(x, 4)).ljust(12) for x in mean)
		print('mean\t|%s' % line)

	# true positive metric
	def tp(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[1, 1]

	# true negative metric
	def tn(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[0, 0]

	# false positive metric
	def fp(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0

		return matrix[0, 1]
	
	# false negative metric
	def fn(self, y_true, y_pred): 
		matrix = confusion_matrix(y_true, y_pred)

		if matrix.shape != (2, 2):
			return 0
		
		return matrix[1, 0]
