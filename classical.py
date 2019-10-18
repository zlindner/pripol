from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import precision_recall_fscore_support, classification_report
from time import time

def svm(params):    
    svm = OneVsRestClassifier(SVC(random_state=42))
    
    return svm

def mnb(params):
    mnb = OneVsRestClassifier(MultinomialNB())

    return mnb

def lr(params):
    lr = OneVsRestClassifier(LogisticRegression(penalty=params['penalty'], tol=params['tol'], C=params['c'], solver='lbfgs', random_state=42))

    return lr

def vectorize_data(x):
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(x)

    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(counts)

    return tfidf

def evaluate(x, y, model_name, params, verbose=True):
    x = vectorize_data(x)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_num = 1
    actual = []
    predictions = []
    total_time = 0

    for train, test in skf.split(x, y):
        if verbose:
            print('fold %s' % fold_num, end='', flush=True)

        fold_num += 1
        fold_start = time()

        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        if model_name == 'svm':
            model = svm(params)
        elif model_name == 'mnb':
            model = mnb(params)
        elif model_name == 'lr':
            model = lr(params)

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        actual.extend(y_test)
        predictions.extend(y_predict)

        fold_end = time()
        fold_time = round(fold_end - fold_start, 2)
        total_time += fold_time

        if verbose:
            print('\t\t%ss' % fold_time)

    results = classification_report(actual, predictions, output_dict=True)

    if verbose:
        print(classification_report(actual, predictions))

    return results, round(total_time, 2)

def tune(x, y, model_name, params):
    grid = list(ParameterGrid(params))

    print('tuning %s with parameters %s' % (model_name, params))

    for params in grid:
        print('testing %s' % params, end='', flush=True)
        
        results, eval_time = evaluate(x, y, model_name, params, verbose=False)
        f = round(results['accuracy'], 2)
        
        print('\t\t%s\t\t%ss' % (f, eval_time))