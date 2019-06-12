from ml.model import Model
from sklearn.linear_model import SGDClassifier

# support vector machine implementation
class SVM(Model):
	
	def __init__(self, opp115):
		self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

		super().__init__(opp115)

	def get_name(self):
		return 'SVM'