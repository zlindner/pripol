from data.opp115 import OPP115
from data.acl1010 import ACL1010
from ml.cnn import CNN
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

opp115 = OPP115()
X_train, X_test, y_train, y_test, max_seq_len = opp115.build_sequences()

#acl1010 = ACL1010()
#vec_acl1010 = acl1010.load_vectors()

# TODO move to opp115.py
y_train = y_train.values.reshape(-1, 1)
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()

cnn = CNN()
model = cnn.build(max_seq_len)
model.fit(X_train, y_train, epochs=2)

print(model.predict(X_test, batch_size=1024))