from ml.model import Model
from sklearn.linear_model import SGDClassifier

# support vector machine implementation
class SVM(Model):
	
	def __init__(self, opp115):
		self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=100, tol=0.001, random_state=42)

		self.params = [{
			'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
			'classifier__max_iter': [100, 250, 500, 1000, 2000],
			'classifier__tol': [None, 0.001, 0.01, 0.1, 1.0],
			'classifier__shuffle': [True, False]
		}]

		super().__init__(opp115)

	def get_name(self):
		return 'SVM'