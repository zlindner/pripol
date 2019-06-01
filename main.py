from data.opp115 import OPP115
from data.acl1010 import ACL1010

opp115 = OPP115()
X_train, X_test, vocab_size = opp115.build_sequences()

acl1010 = ACL1010()
vec_acl1010 = acl1010.load_vectors()