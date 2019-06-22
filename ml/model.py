import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

# TODO add function that formats evaluation output for latex


class Model():

    LABELS = [
        'first_party_collection_use',
        'third_party_sharing_collection',
        'introductory_generic',
        'user_choice_control',
        'international_specific_audiences',
        'data_security',
        'privacy_contact_information',
        'user_access_edit_deletion',
        'practice_not_covered',
        'policy_change',
        'data_retention',
        'do_not_track'
    ]

    def __init__(self, x, y):
        self.binary = None  # TODO needded?

        x, x_test, y, y_test = self.create_test_set(x, y)  # TODO don't split if evluating

        model = self.create()

        self.evaluate(model, x, y)

    def create(self):
        '''Defines the structure of and implements the model'''

        raise NotImplementedError

    def train(self, model, x_train, y_train):
        '''Trains the model with the passed data'''

        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

    def predict(self, model, x_test):
        return model.predict(x_test)

    def tune(self):
        raise NotImplementedError

    def evaluate(self, model, x, y, folds=10):
        '''Evaluates a model using stratified k-fold cross validation'''

        skf = StratifiedKFold(n_splits=folds, random_state=42)

        fold_results = {
            'tn': [],
            'tp': [],
            'fn': [],
            'fp': [],
            'precision': [],
            'recall': [],
            'f': []
        }

        fold = 1  # current fold

        for train, test in skf.split(x, y):
            print('fold %s: ' % fold, end='', flush=True)
            fold += 1

            start = time.time()

            x_train, x_test = x[train], x[test]
            y_train, y_true = y[train], y[test]

            print('training... ', end='', flush=True)
            self.train(model, x_train, y_train)

            print('predicting... ', end='', flush=True)
            y_pred = self.predict(model, x_test)

            tn, tp, fn, fp, precision, recall, f = self.calculate_measures(y_true, y_pred)

            fold_results['tn'].append(sum(tn))  # total fold tn
            fold_results['tp'].append(sum(tp))  # total fold tp
            fold_results['fn'].append(sum(fn))  # total fold fn
            fold_results['fp'].append(sum(fp))  # total fold fp
            fold_results['precision'].append(precision)  # fold precision for each data practice
            fold_results['recall'].append(recall)  # fold recall for each data practice
            fold_results['f'].append(f)  # fold f for each data practice

            end = time.time()
            elapsed = str(np.round(end - start, 2))
            print('finished in %ss' % elapsed)  # elapsed time for each fold

        precision_avg = [round(sum(p) / folds, 2) for p in zip(*fold_results['precision'])]  # average precision for each data practice over folds
        precision_micro = round(sum(fold_results['tp']) / (sum(fold_results['tp']) + sum(fold_results['fp'])), 2)
        p = precision_avg + [precision_micro]

        recall_avg = [round(sum(r) / folds, 2) for r in zip(*fold_results['recall'])]  # average recall for each data practice over folds
        recall_micro = round(sum(fold_results['tp']) / (sum(fold_results['tp']) + sum(fold_results['fn'])), 2)
        r = recall_avg + [recall_micro]

        f_avg = [round(sum(f) / folds, 2) for f in zip(*fold_results['f'])]  # average f for each data practice over folds
        f_micro = round(2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro)), 2)
        f = f_avg + [f_micro]

        l = Model.LABELS + ['micro']
        headers = ['data_practice', 'precision', 'recall', 'f']
        data = [headers] + list(zip(l, p, r, f))

        for i, d in enumerate(data):
            line = ' & '.join(str(x).ljust(32) for x in d)
            print(line)
            if i == 0:
                print(' ' * len(line))

    def create_test_set(self, x, y):
        '''Creates traing and testing subsets by stratified random sampling'''

        return train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

    def calculate_fold_measures(self, y_true, y_pred):
        '''Calculates precision, recall, and f measures for the given predictions'''

        if self.binary:
             # TODO calculations for binary classifier
            pass
        else:
            mcm = multilabel_confusion_matrix(y_true, y_pred)

            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]

        precision = self.precision(tp, fp)
        recall = self.recall(tp, fn)
        f = self.f(precision, recall)

        return tn, tp, fn, fp, precision, recall, f

    def precision(self, tp, fp):
        '''Calculates the precision score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.true_divide(tp, tp + fp)
            p[p == np.inf] = 0
            p = np.nan_to_num(p)

            return p

    def recall(self, tp, fn):
        '''Calculates the recall score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.true_divide(tp, tp + fn)
            r[r == np.inf] = 0
            r = np.nan_to_num(r)

            return r

    def f(self, precision, recall):
        '''Calculates the f score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            f = 2 * np.true_divide(precision * recall, precision + recall)
            f[f == np.inf] = 0
            f = np.nan_to_num(f)

            return f
