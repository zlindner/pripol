from data.opp115 import OPP115
from ml.mnb import MNB
from ml.svm import SVM

opp115 = OPP115()
#opp115.display_statistics()

#mnb = MNB(opp115)
#mnb.cross_validate('kfold')
#mnb.cross_validate('stratified_kfold')

svm = SVM(opp115)
svm.cross_validate('kfold')
#svm.cross_validate('stratified_kfold')
#ssvm.tune_hyperparameters('kfold')