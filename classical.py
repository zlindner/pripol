from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def svm():    
    svm = OneVsRestClassifier(SVC(random_state=42))
    
    return svm

def mnb():
    mnb = OneVsRestClassifier(MultinomialNB())

    return mnb

def lr():
    lr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', random_state=42))

    return lr

def vectorize_data(x):
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(x)

    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)

    return tfidf

def train(model, x_train, y_train):
    model.fit(x_train, y_train)