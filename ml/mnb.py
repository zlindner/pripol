from ml.model import Model
from sklearn.naive_bayes import MultinomialNB

# naive bayes multinomial implementation
class MNB(Model):

	def __init__(self, opp115):
		self.classifier = MultinomialNB(alpha=0.1)

		self.params = [{
			'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
		}]

		super().__init__(opp115)

	def get_name(self):
		return 'MNB'