from data.opp115 import OPP115
from ml.mnb import MNB
from ml.svm import SVM

opp115 = OPP115()

mnb = MNB(opp115)
mnb.kfold()

#svm = SVM(opp115)