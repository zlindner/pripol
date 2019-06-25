import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

# TODO add function that formats evaluation output for latex
# TODO add function that creates table from eval results of all models
# TODO handle nan in binary evaluation measures


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

    def create(self):
        '''Defines the structure of and implements the model'''

        raise NotImplementedError

    def train(self, x_train, y_train):
        '''Trains the model with the passed data'''

        if hasattr(self, 'batch_size') and hasattr(self, 'epochs'):
            self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        else:
            self.model.fit(x_train, y_train)

    def predict(self, x_test):
        '''Predicts y values for the given data'''

        return self.model.predict(x_test)

    def tune(self):
        '''Optimizes the hyperparamaters of the model'''

        raise NotImplementedError

    def evaluate(self, x, y, folds=10):
        '''Evaluates a model using stratified k-fold cross validation'''

        if self.binary:
            self.evaluate_binary(x, y, folds)
        else:
            self.evaluate_multilabel(x, y, folds)

    def evaluate_binary(self, consolidated, encoded, folds):
        skf = StratifiedKFold(n_splits=folds, random_state=42)

        eval_results = []

        for i, label in enumerate(Model.LABELS):
            print('label %s: ' % i, end='', flush=True)

            current_label = encoded[['policy_id', 'segment_id', label]]
            current_label = consolidated.merge(current_label, on=['policy_id', 'segment_id']).drop_duplicates()
            x = current_label['segment']
            y = current_label[label]

            label_results = {
                'tn': 0,
                'tp': 0,
                'fn': 0,
                'fp': 0,
                'precision': [],
                'recall': [],
                'f': []
            }

            print('evaluating... ', end='', flush=True)

            start = time.time()

            for train, test in skf.split(x, y):
                x_train, x_test = x[train], x[test]
                y_train, y_true = y[train], y[test]

                self.train(x_train, y_train)

                y_pred = self.predict(x_test)

                tn, tp, fn, fp, precision, recall, f = self.calc_fold_measures(y_true, y_pred)

                label_results['tn'] += tn
                label_results['tp'] += tp
                label_results['fn'] += fn
                label_results['fp'] += fp
                label_results['precision'].append(precision)  # fold precision for each data practice
                label_results['recall'].append(recall)  # fold recall for each data practice
                label_results['f'].append(f)  # fold f for each data practice

            eval_results.append(label_results)

            end = time.time()
            elapsed = str(np.round(end - start, 2))
            print('finished in %ss' % elapsed)  # elapsed time for each label

        precision, recall, f = self.calc_binary_measures(eval_results, folds)

        self.display_eval_measures(precision, recall, f)

    def evaluate_multilabel(self, x, y, folds):
        skf = StratifiedKFold(n_splits=folds, random_state=42)

        eval_results = {
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
            print('fold %s: evaluating... ' % fold, end='', flush=True)
            fold += 1

            start = time.time()

            x_train, x_test = x[train], x[test]
            y_train, y_true = y[train], y[test]

            self.train(x_train, y_train)

            y_pred = self.predict(x_test)

            tn, tp, fn, fp, precision, recall, f = self.calc_fold_measures(y_true, y_pred)

            eval_results['tn'].append(sum(tn))  # total fold tn
            eval_results['tp'].append(sum(tp))  # total fold tp
            eval_results['fn'].append(sum(fn))  # total fold fn
            eval_results['fp'].append(sum(fp))  # total fold fp
            eval_results['precision'].append(precision)  # fold precision for each data practice
            eval_results['recall'].append(recall)  # fold recall for each data practice
            eval_results['f'].append(f)  # fold f for each data practice

            end = time.time()
            elapsed = str(np.round(end - start, 2))
            print('finished in %ss' % elapsed)  # elapsed time for each fold

        precision, recall, f = self.calc_multilabel_measures(eval_results, folds)

        self.display_eval_measures(precision, recall, f)

    def create_test_set(self, x, y):
        '''Creates traing and testing subsets by stratified random sampling'''

        return train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

    def calc_fold_measures(self, y_true, y_pred):
        '''Calculates precision, recall, and f measures for the given predictions'''

        if self.binary:
            cm = confusion_matrix(y_true, y_pred)

            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
        else:
            mcm = multilabel_confusion_matrix(y_true, y_pred)

            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]

            # sharcnet
            # cm = confusion_matrix(y_true, y_pred)
            # fp = cm.sum(axis=0) - np.diag(cm)
            # fn = cm.sum(axis=1) - np.diag(cm)
            # tp = np.diag(cm)
            # tn = cm.sum() - (fp + fn + tp)

        precision = self.precision(tp, fp)
        recall = self.recall(tp, fn)
        f = self.f(precision, recall)

        return tn, tp, fn, fp, precision, recall, f

    def calc_binary_measures(self, eval_results, folds):
        '''Calculates a binary evaluation's micro average precision, recall, and f measures'''

        tp = [label['tp'] for label in eval_results]
        fp = [label['fp'] for label in eval_results]
        fn = [label['fn'] for label in eval_results]

        precision_avg = [round(sum(label['precision']) / folds, 2) for label in eval_results]
        precision_micro = round(sum(tp) / (sum(tp) + sum(fp)), 2)
        precision = precision_avg + [precision_micro]

        recall_avg = [round(sum(label['recall']) / folds, 2) for label in eval_results]
        recall_micro = round(sum(tp) / (sum(tp) + sum(fn)), 2)
        recall = recall_avg + [recall_micro]

        f_avg = [round(sum(label['f']) / folds, 2) for label in eval_results]
        f_micro = round(2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro)), 2)
        f = f_avg + [f_micro]

        return precision, recall, f

    def calc_multilabel_measures(self, eval_results, folds):
        '''Calculates a multilabel evaluation's micro average precision, recall, and f measures'''

        precision_avg = [round(sum(p) / folds, 2) for p in zip(*eval_results['precision'])]  # average precision for each data practice over folds
        precision_micro = round(sum(eval_results['tp']) / (sum(eval_results['tp']) + sum(eval_results['fp'])), 2)
        precision = precision_avg + [precision_micro]

        recall_avg = [round(sum(r) / folds, 2) for r in zip(*eval_results['recall'])]  # average recall for each data practice over folds
        recall_micro = round(sum(eval_results['tp']) / (sum(eval_results['tp']) + sum(eval_results['fn'])), 2)
        recall = recall_avg + [recall_micro]

        f_avg = [round(sum(f) / folds, 2) for f in zip(*eval_results['f'])]  # average f for each data practice over folds
        f_micro = round(2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro)), 2)
        f = f_avg + [f_micro]

        return precision, recall, f

    def display_eval_measures(self, precision, recall, f):
        '''Displays evaluation metrics for each data practice as a table'''

        labels = Model.LABELS + ['micro']
        headers = ['data_practice', 'precision', 'recall', 'f']
        data = [headers] + list(zip(labels, precision, recall, f))

        for i, d in enumerate(data):
            line = ' | '.join(str(x).ljust(32) for x in d)
            print(line)
            if i == 0:
                print(' ' * len(line))

    def precision(self, tp, fp):
        '''Calculates the precision score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.true_divide(tp, tp + fp)

            if self.binary:
                return p

            p[p == np.inf] = 0
            p = np.nan_to_num(p)

            return p

    def recall(self, tp, fn):
        '''Calculates the recall score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.true_divide(tp, tp + fn)

            if self.binary:
                return r

            r[r == np.inf] = 0
            r = np.nan_to_num(r)

            return r

    def f(self, precision, recall):
        '''Calculates the f score(s) for a confusion matrix'''

        with np.errstate(divide='ignore', invalid='ignore'):
            f = 2 * np.true_divide(precision * recall, precision + recall)

            if self.binary:
                return f

            f[f == np.inf] = 0
            f = np.nan_to_num(f)

            return f
