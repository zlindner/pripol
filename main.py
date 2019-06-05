from data.opp115 import OPP115
from data.acl1010 import ACL1010
from ml.cnn import CNN
import pandas as pd
import numpy as np
from keras import layers
from keras.models import Sequential
from gensim.models import KeyedVectors as kv

EMBEDDING_DIM = 300

opp115 = OPP115()
acl1010 = ACL1010()

cnn = CNN(opp115, acl1010)
model = cnn.build()
model.fit(opp115.X_train, opp115.y_train, epochs=1, batch_size=1024)
pred = model.predict(opp115.X_test)

df = pd.DataFrame(columns=['segment'] + opp115.DATA_PRACTICES)
df['segment'] = opp115.tokenizer.sequences_to_texts(opp115.X_test)
#df['data_practice'] = opp115.encoder.inverse_transform(opp115.y_test).ravel() # flatten()?
#df['data_practice'] = np.argmax(opp115.y_test)
df[opp115.DATA_PRACTICES] = pred

print(df.head(20))
#df.to_csv('results.csv')

#model.fit(opp115.X_train, opp115.y_train, epochs=1, validation_data=[opp115.X_test, opp115.y_test])

#oss, accuracy = model.evaluate(opp115.X_train, opp115.y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
#loss, accuracy = model.evaluate(opp115.X_test, opp115.y_test, verbose=False)
#print("Testing Accuracy:  {:.4f}".format(accuracy))

#pred = np.round(pred, 5)