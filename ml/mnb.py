from ml.model import Model
from sklearn.naive_bayes import MultinomialNB

# naive bayes multinomial implementation
class MNB(Model):

	def __init__(self, opp115):
		self.classifier = MultinomialNB(alpha=1e-3)

		super().__init__(opp115)

	def get_name(self):
		return 'MNB'