import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix, classification_report, precision_recall_fscore_support

class Model():

	LABELS = [
			'first_party_collection_use',
			'third_party_sharing_collection', 
			'introductory_generic', 
			'user_choice_control', 
			'international_specific_audiences', 
			'data_security', 
			'privacy_contact_information', 
			'user_access_edit_deletion', 
			'practice_not_covered', 
			'policy_change', 
			'data_retention', 
			'do_not_track'
	]

	def __init__(self, x, y):	
		self.binary = None # TODO needded?

		x, x_test, y, y_test = self.create_test_set(x, y)

		model = self.create()

		skf = StratifiedKFold(n_splits=10, random_state=42)
		fold = 1
		
		results = {
			
		}

		for train, test in skf.split(x, y):
			print('fold %s: ' % fold, end='', flush=True)
			fold += 1

			start = time.time()
			
			x_train, x_test = x[train], x[test]
			y_train, y_true = y[train], y[test]

			print('training... ', end='', flush=True)
			self.train(model, x_train, y_train)

			print('predicting... ', end='', flush=True)
			y_pred = self.predict(model, x_test)

			# TODO calculations for binary classifier
			mcm = multilabel_confusion_matrix(y_true, y_pred)
			
			tn = mcm[:, 0, 0]
			tp = mcm[:, 1, 1]
			fn = mcm[:, 1, 0]
			fp = mcm[:, 0, 1]

			precision = self.precision(tp, fp)
			recall = self.recall(tp, fn)
			f = self.f(precision, recall)
			
			end = time.time()
			elapsed = str(np.round(end - start, 2))

			print('finished in %ss' % elapsed)

		print(results)

	# defines and creates the model
	def create(self):
		raise NotImplementedError

	# trains the model
	def train(self, model, x_train, y_train):
		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

	# TODO comment
	def predict(self, model, x_test):
		return model.predict(x_test)

	# tunes the model's hyperparameters
	def tune(self):
		raise NotImplementedError

	# evalutates the model by cross validation 
	def evaluate(self, model, x, y, metrics):
		scores = cross_validate(estimator=model, X=x, y=y, cv=10, scoring=metrics)

		print(scores)

		results = {
			'precision': round(np.mean(scores['test_precision']), 2),
			'recall': round(np.mean(scores['test_recall']), 2),
			'f1': round(np.mean(scores['test_f1']), 2),
		}

		return results

	# creates traing and testing subsets by stratified random sampling
	def create_test_set(self, x, y):
		return train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

	def precision(self, tp, fp):
		'''Calculates the precision score(s) for a confusion matrix

		Args:
			tp: number of true positive predictions
			fp: number of false positive predicitions

		Returns:
			The precision score(s) for the confusion matrix 
		'''

		with np.errstate(divide='ignore', invalid='ignore'):
			p = np.true_divide(tp, tp + fp)
			p[p == np.inf] = 0
			p = np.nan_to_num(p)

			return p

	def recall(self, tp, fn):
		'''Calculates the recall score(s) for a confusion matrix

		Args:
			tp: number of true positive predictions
			fn: number of false negative predicitions

		Returns:
			The recall score(s) for the confusion matrix 
		'''

		with np.errstate(divide='ignore', invalid='ignore'):
			r = np.true_divide(tp, tp + fn)
			r[r == np.inf] = 0
			r = np.nan_to_num(r)

			return r

	def f(self, precision, recall):
		'''Calculates the f score(s) for a confusion matrix

		Args:
			precision: the precision score(s)
			recall: the recall score(s)

		Returns:
			The f scores(s) for the confusion matrix 
		'''

		with np.errstate(divide='ignore', invalid='ignore'):
			f = 2 * np.true_divide(precision * recall, precision + recall)
			f[f == np.inf] = 0
			f = np.nan_to_num(f)

			return f