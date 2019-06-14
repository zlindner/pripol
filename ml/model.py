import numpy as np
import warnings

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning

# superclass for sklearn models
class Model():

	def __init__(self, opp115):
		print('\n# %s' % self.get_name())

		self.opp115 = opp115
		self.labels = ['first_party_collection_use', 'third_party_sharing_collection', 'introductory_generic', 'user_choice_control', 'international_specific_audiences', 'data_security', 'privacy_contact_information', 'user_access_edit_deletion', 'practice_not_covered', 'policy_change', 'data_retention', 'do_not_track']
		#self.labels = ['policy_change', 'first_party_collection_use', 'third_party_sharing_collection', 'do_not_track', 'user_choice_control', 'international_specific_audiences', 'data_security', 'data_retention', 'user_access_edit_deletion', 'introductory_generic', 'privacy_contact_information', 'practice_not_covered']

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

		self.kfold = KFold(n_splits=10, random_state=42)
		self.stratified_kfold = StratifiedKFold(n_splits=10, random_state=42)

		# disable UndefinedMetricWarning
		warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

	# returns the name of the model
	def get_name(self):
		raise NotImplementedError

	def cross_validate(self, strategy_name):
		if strategy_name == 'basic':
			strategy = self.train_test()
		elif strategy_name == 'kfold':
			strategy = self.kfold
		elif strategy_name == 'stratified_kfold':
			strategy = self.stratified_kfold
		else:
			print('Incorrect cross_validate strategy supplied, options include basic, kfold, stratified_kfold')
			return

		results = {}

		for label in self.labels:
			target = self.opp115.encoded[['policy_id', 'segment_id', label]]
			target = self.opp115.consolidated.merge(target, on=['policy_id', 'segment_id']).drop_duplicates()
			x = target['segment']
			y = target[label]

			scores = cross_validate(estimator=self.model, X=x, y=y, cv=strategy, scoring=self.metrics)

			results[label] = {
				'precision': round(np.mean(scores['test_precision']), 2),
				'recall': round(np.mean(scores['test_recall']), 2),
				'f1': round(np.mean(scores['test_f1']), 2),
				'tp': sum(scores['test_tp']),
				'tn': sum(scores['test_tn']),
				'fp': sum(scores['test_fp']),
				'fn': sum(scores['test_fn']),

				'fold_precision': scores['test_precision'],
				'fold_recall': scores['test_recall'],
				'fold_f1': scores['test_f1'],
				'fold_tp': scores['test_tp'],
				'fold_tn': scores['test_tn'],
				'fold_fp': scores['test_fp'],
				'fold_fn': scores['test_fn']
			}

		self.display_results(results, strategy_name)

	# basic train / test subset splitting TODO needed?
	def train_test(self):
		pass

	# tune hyperparameters for model
	def tune_hyperparameters(self, strategy_name):
		if strategy_name == 'basic':
			strategy = self.train_test()
		elif strategy_name == 'kfold':
			strategy = self.kfold
		elif strategy_name == 'stratified_kfold':
			strategy = self.stratified_kfold
		else:
			print('Incorrect cross_validate strategy supplied, options include basic, kfold, stratified_kfold')
			return

		print('tuning hyperparameters...')

		clf = GridSearchCV(self.model, self.params, cv=strategy)

		for label in self.labels:
			target = self.opp115.encoded[['policy_id', 'segment_id', label]]
			target = self.opp115.consolidated.merge(target, on=['policy_id', 'segment_id']).drop_duplicates()
			x = target['segment']
			y = target[label]

			clf.fit(x, y)

			print('best parameters set found on development set')
			print(clf.best_params_)

			break # ???

	# displays metrics (precision, recall, f) from cross validation
	def display_results(self, results, strategy_name):
		print('\n%s' % strategy_name)

		micro_precision, micro_recall, micro_f = self.calc_micro(results)

		headers = ['label', 'precision', 'recall', 'f1']
		labels = list(results.keys()) + ['micro']
		precision = [list(results[x].values())[0] for x in results] + [micro_precision]
		recall = [list(results[x].values())[1] for x in results] + [micro_recall]
		f1 = [list(results[x].values())[2] for x in results] + [micro_f]

		data = [headers] + list(zip(labels, precision, recall, f1))

		for i, d in enumerate(data):
			line = '|'.join(str(x).ljust(32) for x in d)
			print(line)
			if i == 0:
				print('-' * len(line))

	# calculates micro average for precision, recall, and f between data practices
	def calc_micro(self, results):
		tp = [list(results[x].values())[3] for x in results]
		tn = [list(results[x].values())[4] for x in results]
		fp = [list(results[x].values())[5] for x in results]
		fn = [list(results[x].values())[6] for x in results]

		micro_precision = sum(tp) / (sum(tp) + sum(fp))
		micro_recall = sum(tp) / (sum(tp) + sum(fn))
		micro_f = 2 * (micro_precision * micro_recall / (micro_precision + micro_recall))

		return round(micro_precision, 2), round(micro_recall, 2), round(micro_f, 2)

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
