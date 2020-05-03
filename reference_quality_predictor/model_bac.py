# -*- coding: utf-8 -*-
"""
Created on May 1 2017

@author: Alessandro
"""

import os
import sys

import argparse

reload(sys)
sys.setdefaultencoding("utf8")

import pandas as pd
import scipy as sp
import numpy as np
import chardet
import re

from joblib import dump, load

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB


####starts HERE
### Predict relevance

###baseline
#
#
# path = '/Users/alessandro/Documents/PhD/WD_refs_results/baseline.csv'
#
# baseline = pd.read_csv(path)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


###confusion matrix counter
def conf_counter(y_pred, y_test):
    y_pred = list(y_pred)
    y_test = list(y_test)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):

        if y_pred[i] == 1 and y_test[i] == 1:
            TP += 1

        elif y_pred[i] == 0 and y_test[i] == 1:
            FN += 1

        elif y_pred[i] == 1 and y_test[i] == 0:
            FP += 1

        elif y_pred[i] == 0 and y_test[i] == 0:
            TN += 1

    return TP, FP, FN, TN


###f1 score compute
def f1_compute(tp_list, fp_list, fn_list):
    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)

    f1_score_custom = (2 * tp) / float(2 * tp + fp + fn)

    return f1_score_custom


def dataset_preprocess(prediction_column, train_data, test_data, cv=False):
    # train_data = os.path.realpath('./results/train_set_references.csv')
    # train_data = '/Users/alessandro/Documents/WD_references_analysis/results/prediction_data.csv'
    # train_df = pd.read_csv(train_data, sep='\t', header=0)
    train_df = pd.read_csv(train_data)
    # Create vectorizer for function to use
    # vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
    item_vectorizer = TfidfVectorizer()
    object_vectorizer = TfidfVectorizer()
    vec = DictVectorizer()

    if test_data:
        test_df = pd.read_csv(test_data)
        columns = ['item_data', 'object_data', 'stat_property', 'user_type', 'http_code', 'user_ref_edits',
                   'user_edits', 'url_use', 'domain_use']
        all_data = pd.concat([train_df[columns], test_df[columns]])
        # Z
        vec.fit(all_data[['stat_property',
                          'user_type',
                          'http_code']].to_dict(orient='records'))

        # item_vec = item_vectorizer.transform(train_df['item_data'])
        # item_vocab = item_vectorizer.vocabulary_.items()
        # object_vec = object_vectorizer.transform(train_df['object_data'])
        # object_vocab = object_vectorizer.vocabulary_.items()
        vec_features = vec.transform(train_df[['stat_property',
                                               'user_type',
                                               'http_code']].to_dict(orient='records')).toarray()
        X_data = sp.sparse.hstack((vec_features,
                                   train_df[['user_edits', 'user_ref_edits', 'url_use', 'domain_use']].values),
                                  format='csr')

        # item_vec_test = item_vectorizer.transform(test_df['item_data'])
        # object_vec_test = object_vectorizer.transform(test_df['object_data'])
        vec_features_test = vec.transform(test_df[['stat_property',
                                             'user_type',
                                             'http_code']].to_dict(orient='records')).toarray()
        X_test = sp.sparse.hstack((vec_features_test,
                                   test_df[['user_edits', 'user_ref_edits', 'url_use', 'domain_use']].values),
                                  format='csr')
        # X_data = train_df[['user_edits', 'user_ref_edits', 'url_use', 'domain_use',
        #                                      'user_type',
        #                                      'http_code']]
        # X_test = test_df[['user_edits', 'user_ref_edits', 'url_use', 'domain_use',
        #                                     'user_type',
        #                                     'http_code']]

        y_train = train_df[prediction_column]
        y_test = test_df[prediction_column]


    else:
        # item_vec = item_vectorizer.fit_transform(train_df['item_data'])
        # item_vocab = item_vectorizer.vocabulary_.items()
        # object_vec = object_vectorizer.fit_transform(train_df['object_data'])
        # object_vocab = object_vectorizer.vocabulary_.items()
        vec_features = vec.fit_transform(train_df[['stat_property',
                                             'user_type',
                                             'http_code']].to_dict(orient='records')).toarray()
        X_data = sp.sparse.hstack((vec_features,
                                   train_df[['user_edits', 'user_ref_edits', 'url_use', 'domain_use',
                                             'user_type',
                                             'http_code']].values),
                                  format='csr')

    # rev_dictionary = {v: k for k, v in item_vocab}
    # column_names_from_item_features = [v for k, v in rev_dictionary.items()]
    # rev_dictionary = {v: k for k, v in object_vocab}
    # column_names_from_object_features = [v for k, v in rev_dictionary.items()]
    #
    data_columns = vec.feature_names_ + [
        'user_edits', 'user_ref_edits',
        'url_use', 'domain_use']
    # data_columns = ['user_edits', 'user_ref_edits',
    #                 'url_use', 'domain_use', 'user_type', 'http_code']

    if cv:
        y = train_df[prediction_column]

        return X_data, y, data_columns
    else:
        # prediction_column = raw_input('Type authoritative or support_object:\n')
        if test_data:
            return X_data, X_test, y_train, y_test, data_columns

        X_train, X_test, y_train, y_test = train_test_split(X_data, train_df[prediction_column],
                                                            test_size=0.3,
                                                            random_state=53)

        return X_train, X_test, y_train, y_test, data_columns


### Model training
class modelTrainer(object):

    def __init__(self, train_data, test_data):
        # values = {'authoritativeness': "authoritative",
        #           'relevance': "support_object"}
        models = {'baseline': self.baseline, 'svm_model': self.svm_model, 'svm_model_cv': self.svm_model_cv,
                  'linear_svm': self.linear_svm, 'linear_svm_cv': self.linear_svm_cv,
                  'rf_model': self.rf_model, 'rf_model_cv': self.rf_model_cv,
                  'nb_model': self.nb_model, 'nb_model_cv': self.nb_model_cv}
        self.means = raw_input("Which model do you want to train? "
                               "(Choices: 'baseline', 'svm_model', 'svm_model_cv', 'rf_model', rf_model_cv', 'nb_model', 'nb_model_cv', 'all_cv')\n")
        prediction_col = raw_input(
            "Please choose the attribute you want to evaluate (enter 0 for 'authoritativeness', 1 for 'relevance'):")
        prediction_goal = {'0': 'authoritative',
                           '1': 'support_object'}
        self.prediction_column = prediction_goal[prediction_col]
        self.train_data = train_data
        self.test_data = test_data
        if self.means != 'all_cv':
            models[self.means]()
        else:
            self.baseline()
            self.nb_model_cv()
            self.rf_model_cv()
            self.svm_model_cv()

    def baseline(self):
        # train_data = '/Users/alessandro/Documents/WD_references_analysis/results/prediction_data.csv'
        train_data = os.path.realpath('./results/prediction_data.csv')
        posts = pd.read_csv(train_data, sep='\t', header=0)

        if self.prediction_column == 'authoritative':
            predicted = posts['authoritative_baseline']
            expected = posts['authoritative']
        elif self.prediction_column == 'support_object':
            predicted = posts['statement_match']
            expected = posts['support_object']

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "Baseline precision:{0}; recall:{1}; f1:{2}; auc_pr:{3}; mcc:{4}".format(str(precision), str(recall),
                                                                                       str(f1), str(auc_pr), str(mcc))
        file_name = 'baseline_results_' + str(self.prediction_column) + '.csv'
        with open(file_name, 'w') as f:
            f.write("Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))

    def svm_model(self):
        print 'you chose SVM'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

        ###SVM
        clf = svm.SVC(kernel='rbf', C=0.4, cache_size=1000, class_weight='balanced').fit(X_train, y_train)

        ###compute scores
        clf.score(X_test, y_test)
        predicted = clf.predict(X_test)
        expected = y_test

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
            f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'svm_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))

    ###RF cross validation
    def svm_model_cv(self):
        print 'you chose SVM cross_validation'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data, cv=True)
        print 'data processed'

        X_train = data[0]
        y_test = data[1]
        label_type = str(self.prediction_column)

        clf = svm.SVC(kernel='rbf', C=0.4, cache_size=1000, class_weight='balanced')
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        f1_list = []
        auc_pr_list = []
        mcc_list = []
        true_positive_list = []
        false_positive_list = []
        false_negative_list = []

        for train_index, test_index in crossvalidation.split(data[0], data[1]):
            X_train, X_test = data[0][train_index], data[0][test_index]
            y_train, y_test = data[1][train_index], data[1][test_index]

            clf = clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)

            expected = y_test
            conf_scores = conf_counter(predicted, expected)
            true_positive = conf_scores[0]
            true_positive_list.append(true_positive)
            false_positive = conf_scores[1]
            false_positive_list.append(false_positive)
            false_negative = conf_scores[2]
            false_negative_list.append(false_negative)

            precision = precision_score(expected, predicted, average='weighted', pos_label=1)
            precision_list.append(precision)
            recall = recall_score(expected, predicted, average='weighted', pos_label=1)
            recall_list.append(recall)
            f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
            f1_list.append(f1)
            auc_pr = average_precision_score(expected, predicted, average='weighted')
            auc_pr_list.append(auc_pr)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)

        print "SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(
            mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
            mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(
                mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###linear SVM
    def linear_svm(self):
        print 'you chose linear SVM'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

        ###SVM
        clf = svm.LinearSVC(C=0.5, class_weight='balanced').fit(X_train, y_train)

        ###compute scores
        clf.score(X_test, y_test)
        predicted = clf.predict(X_test)
        expected = y_test

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "Linear SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
            f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'linear_svm_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("Linear SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))

    def linear_svm_cv(self):
        print 'you chose linear SVM cross_validation'
        data = dataset_preprocess(self.prediction_column, self.train_data, cv=True)
        print 'data processed'

        X_train = data[0]
        y_test = data[1]
        label_type = str(self.prediction_column)

        from sklearn import svm
        from sklearn.model_selection import StratifiedKFold
        clf = svm.LinearSVC(C=0.5, class_weight='balanced')
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        f1_list = []
        auc_pr_list = []
        mcc_list = []
        true_positive_list = []
        false_positive_list = []
        false_negative_list = []

        for train_index, test_index in crossvalidation.split(data[0], data[1]):
            X_train, X_test = data[0][train_index], data[0][test_index]
            y_train, y_test = data[1][train_index], data[1][test_index]

            clf = clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)

            expected = y_test
            conf_scores = conf_counter(predicted, expected)
            true_positive = conf_scores[0]
            true_positive_list.append(true_positive)
            false_positive = conf_scores[1]
            false_positive_list.append(false_positive)
            false_negative = conf_scores[2]
            false_negative_list.append(false_negative)

            precision = precision_score(expected, predicted, average='weighted', pos_label=1)
            precision_list.append(precision)
            recall = recall_score(expected, predicted, average='weighted', pos_label=1)
            recall_list.append(recall)
            f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
            f1_list.append(f1)
            auc_pr = average_precision_score(expected, predicted, average='weighted')
            auc_pr_list.append(auc_pr)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)

        print "Linear SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(
            mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
            mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'linear_svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("Linear SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(
                mean(auc_pr_list)) + "; mcc:" + str(
                mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###RF
    def rf_model(self):
        print 'you chose RF'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        data_columns = data[4]
        label_type = str(self.prediction_column)

        clf = RandomForestClassifier(n_estimators=2000, max_depth=None, min_samples_split=3, random_state=0,
                                     oob_score=True)
        clf = clf.fit(X_train, y_train)
        feature_importances = pd.DataFrame(clf.feature_importances_,
                                           index=data_columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        feature_importances.to_csv('./rf_feature_importances_{}.csv'.format(label_type))
        print(clf.oob_score_)

        predicted = clf.predict(X_test)

        expected = y_test

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
            f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'rf_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))
        # save model to file
        dump(clf, 'rf_model_{}.joblib'.format(label_type))

    ###RF cross validation
    def rf_model_cv(self):
        print 'you chose RF cross_validation'
        data = dataset_preprocess(self.prediction_column, self.train_data, cv=True)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(self.prediction_column)

        clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        auc_pr_list = []
        mcc_list = []
        f1_list = []
        true_positive_list = []
        false_positive_list = []
        false_negative_list = []
        feature_importances_list = []
        data_columns = data[2]
        for train_index, test_index in crossvalidation.split(data[0], data[1]):
            X_train, X_test = data[0][train_index], data[0][test_index]
            y_train, y_test = data[1][train_index], data[1][test_index]

            clf = clf.fit(X_train, y_train)
            predicted = clf.predict(X_test)
            feature_importances_list.append(pd.DataFrame(clf.feature_importances_,
                                                         index=data_columns,
                                                         columns=['importance']).sort_values('importance',
                                                                                             ascending=False))

            expected = y_test

            conf_scores = conf_counter(predicted, expected)
            true_positive = conf_scores[0]
            true_positive_list.append(true_positive)
            false_positive = conf_scores[1]
            false_positive_list.append(false_positive)
            false_negative = conf_scores[2]
            false_negative_list.append(false_negative)

            precision = precision_score(expected, predicted, average='weighted', pos_label=1)
            precision_list.append(precision)
            recall = recall_score(expected, predicted, average='weighted', pos_label=1)
            recall_list.append(recall)
            f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
            f1_list.append(f1)
            auc_pr = average_precision_score(expected, predicted, average='weighted')
            auc_pr_list.append(auc_pr)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)
        feature_importances = pd.concat(feature_importances_list)
        feature_importances.to_csv('./rf_feature_importances_{}.csv'.format(label_type))

        print "RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
            mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
            mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'rf_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(
                mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))
        # save model to file
        dump(clf, 'rf_model_{}.joblib'.format(label_type))

    ###Naive Bayes
    def nb_model(self):
        print 'you chose NB'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

        ber = BernoulliNB()
        # y_pred = gnb.fit(X_train.toarray(), y_train).predict(X_test.toarray())
        clf = ber.fit(X_train.toarray(), y_train)
        predicted = clf.predict(X_test.toarray())
        expected = y_test
        # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
            f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'NB_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))

    ###Naive Bayes cross validation
    def nb_model_cv(self):
        print 'you chose NB cross_validation'
        data = dataset_preprocess(self.prediction_column, self.train_data, cv=True)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(self.prediction_column)

        from sklearn.naive_bayes import BernoulliNB
        # gnb = GaussianNB()
        ber = BernoulliNB()
        # y_pred = gnb.fit(X_train.toarray(), y_train).predict(X_test.toarray())
        # clf = ber#.fit(X_train.toarray(), y_test)
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        f1_list = []
        auc_pr_list = []
        mcc_list = []
        true_positive_list = []
        false_positive_list = []
        false_negative_list = []

        for train_index, test_index in crossvalidation.split(data[0], data[1]):
            X_train, X_test = data[0][train_index], data[0][test_index]
            y_train, y_test = data[1][train_index], data[1][test_index]

            clf = ber.fit(X_train.toarray(), y_train)
            predicted = clf.predict(X_test.toarray())

            expected = y_test
            conf_scores = conf_counter(predicted, expected)
            true_positive = conf_scores[0]
            true_positive_list.append(true_positive)
            false_positive = conf_scores[1]
            false_positive_list.append(false_positive)
            false_negative = conf_scores[2]
            false_negative_list.append(false_negative)

            precision = precision_score(expected, predicted, average='weighted', pos_label=1)
            precision_list.append(precision)
            recall = recall_score(expected, predicted, average='weighted', pos_label=1)
            recall_list.append(recall)
            f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
            f1_list.append(f1)
            auc_pr = average_precision_score(expected, predicted, average='weighted')
            auc_pr_list.append(auc_pr)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)

        print "NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
            mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
            mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'NB_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(
                mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))

    # def function_chooser(self):
    #
    #     method = getattr(self.means)
    #     method()
    #     new_string = 'method called ' + self.means
    #     print new_string


def parse_args():
    parser = argparse.ArgumentParser(description='Training dataset')
    train_data = os.path.realpath('./results/prediction_data.csv')
    parser.add_argument('--train_data', default=train_data,
                        help='Train dataset path')
    parser.add_argument('--test_data', default=None,
                        help='Test dataset path')
    return parser.parse_args()


def main():
    args = parse_args()
    modelTrainer(args.train_data, args.test_data)


if __name__ == "__main__":
    main()
