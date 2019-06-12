import pandas as pd
import numpy as np
from data.opp115 import OPP115
from data.acl1010 import ACL1010
from ml.cnn import CNN

# svm binary for each datapractice start w/ first party
# nbm
# statistics for opp115
# a, b, c, d, precision, recall, fscore, micro + macro averages
# deeplearning.ai cnn videos

opp115 = OPP115()
acl1010 = ACL1010()

cnn = CNN(opp115, acl1010)
model = cnn.build()
model.fit(opp115.X_train, opp115.y_train, epochs=1, batch_size=50)
pred = model.predict(opp115.X_test)
#pred[pred >= 0.5] = 1
#pred[pred < 0.5] = 0

df = pd.DataFrame(columns=['segment', 'data_practice'] + opp115.DATA_PRACTICES)
df['segment'] = opp115.tokenizer.sequences_to_texts(opp115.X_test)
df['data_practice'] = opp115.encoder.inverse_transform(opp115.y_test).ravel() # flatten()?
df[opp115.DATA_PRACTICES] = pred

print(df[['data_practice'] + opp115.DATA_PRACTICES].head(20))
#df.to_csv('results.csv')

#model.fit(opp115.X_train, opp115.y_train, epochs=1, validation_data=[opp115.X_test, opp115.y_test])

#oss, accuracy = model.evaluate(opp115.X_train, opp115.y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
#loss, accuracy = model.evaluate(opp115.X_test, opp115.y_test, verbose=False)
#print("Testing Accuracy:  {:. 4f}".format(accuracy))