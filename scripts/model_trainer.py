# -*- coding: utf-8 -*-
"""
Created on May 1 2017

@author: Alessandro
"""

import os
import sys

reload(sys)
sys.setdefaultencoding("utf8")
# import urllib2
# from bs4 import BeautifulSoup
import pandas as pd
# import re, string
# import ngram
# import nltk
import scipy as sp
import numpy as np
import chardet
# import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

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

def dataset_preprocess(prediction_column):
    train_data =  './prediction_data.csv' #'./prediction_no_census.csv'
    posts = pd.read_csv(train_data, sep='\t', header=0)
    # Create vectorizer for function to use
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    vec = DictVectorizer()
    # y = posts["user_edits"].values.astype(np.float32)

    # X_data = vec.fit_transform(posts[['ref_value', 'ref_domain', 'stat_property', 'stat_value', 'item_id', 'user_type', 'user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].to_dict(orient = 'records')).toarray()
    X_data = sp.sparse.hstack((vectorizer.fit_transform(posts.item_text_clean), vectorizer.fit_transform(posts.object_text), vec.fit_transform(posts[['stat_property', 'user_type', 'code_2',  'instance_of', 'subclass', 'object_instance_of', 'object_subclass', 'property_instance_of']].to_dict(orient='records')).toarray(),posts[['user_edits', 'user_ref_edits_pc', 'ref_count', 'domain_count']].values),format='csr')
    # X_data = vectorizer.fit_transform(posts.item_text)
    # X_columns=vectorizer.get_feature_names()+vec.get_feature_names()+posts[['user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].columns.tolist()
    # X_data = pd.get_dummies(posts[['subclass']])

    prediction_column = raw_input('authoritative or support_object:\n')
    # X_train, X_test, y_train, y_test = train_test_split(posts[['user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].values, posts[prediction_column], test_size=0.3,random_state=47)
    X_train, X_test, y_train, y_test = train_test_split(X_data, posts[prediction_column], test_size=0.3, random_state=53)

    return X_train, X_test, y_train, y_test

def dataset_preprocess_cv(prediction_column):
    train_data = './prediction_data.csv' #'./prediction_no_census.csv'
    posts = pd.read_csv(train_data, sep='\t', header=0)
    # Create vectorizer for function to use
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    vec = DictVectorizer()
    # y = posts["user_edits"].values.astype(np.float32)

    # X_data = vec.fit_transform(posts[['ref_value', 'ref_domain']].to_dict(orient = 'records')).toarray()
    X_data = sp.sparse.hstack((vectorizer.fit_transform(posts.item_text_clean), vectorizer.fit_transform(posts.object_text), vec.fit_transform(posts[['stat_property', 'stat_value', 'user_type',  'instance_of', 'subclass', 'object_instance_of', 'object_subclass', 'property_instance_of']].to_dict(orient='records')).toarray(),posts[['user_edits', 'user_ref_edits_pc', 'ref_count', 'domain_count']].values),format='csr')
    # X_data = vectorizer.fit_transform(posts.item_text)
    # X_columns=vectorizer.get_feature_names()+vec.get_feature_names()+posts[['user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].columns.tolist()
    # X_data = pd.get_dummies(posts[['stat_property', 'stat_value', 'user_type', 'code_2', 'instance_of', 'subclass','object_instance_of', 'object_subclass', 'property_instance_of']])
    prediction_column = raw_input('authoritative or support_object:\n')
    # X_train, X_test, y_train, y_test = train_test_split(posts[['user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].values, posts[prediction_column], test_size=0.3,random_state=47)
    #X_train, X_test, y_train, y_test = train_test_split(X_data, posts[prediction_column], test_size=0.3,random_state=53)
    y = posts[prediction_column]

    return X_data, y


###split train test
# X_train, X_test = train_test_split(posts,  test_size=0.33, random_state=42)
class test:

    def __init__(self, means=None):
        self.means = means


    def baseline(prediction_column):
        train_data = './prediction_data.csv' 
        posts = pd.read_csv(train_data, sep='\t', header=0)

        prediction_column = raw_input("authority_baseline or relevance_match?\n")

        if prediction_column == 'authority_baseline':
            expected_column = 'authoritative'
        elif prediction_column == 'relevance_match':
            prediction_column = 'statement_match'
            expected_column = 'support_object'

        predicted = posts[prediction_column]
        expected = posts[expected_column]

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'baseline_results_' + str(prediction_column) + '.csv'
        with open(file_name, 'w') as f:
            f.write("Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))




    def svm_model(prediction_column):
        print 'you chose SVM'
        data = dataset_preprocess(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(prediction_column)

        ###SVM
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
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

        print "SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'svm_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))

    ###RF cross validation
    def svm_model_cv(prediction_column):
        print 'you chose SVM cross_validation'
        data = dataset_preprocess_cv(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_test = data[1]
        label_type = str(prediction_column)

        from sklearn import svm
        #from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import StratifiedKFold
        clf = svm.SVC(kernel='rbf', C=0.4, cache_size=1000, class_weight='balanced')#.fit(X_train, y_train)

        # crossvalidation = KFold(n_splits=10, shuffle=True, random_state=3)
        # score_type = raw_input('precision, recall, roc_auc, f1, or matthews_corrcoef:\n')
        # scores = cross_val_score(clf, X_train, y_test, scoring=score_type, cv=crossvalidation, n_jobs=4)
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

        print "SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###linear SVM
    def linear_svm(prediction_column):
        print 'you chose linear SVM'
        data = dataset_preprocess(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(prediction_column)

        ###SVM
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
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

    def linear_svm_cv(prediction_column):
        print 'you chose linear SVM cross_validation'
        data = dataset_preprocess_cv(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_test = data[1]
        label_type = str(prediction_column)

        from sklearn import svm
        # from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import StratifiedKFold
        clf = svm.LinearSVC(C=0.5, class_weight='balanced')  # .fit(X_train, y_train)
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
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
                mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###RF
    def rf_model(prediction_column):
        print 'you chose RF'
        data = dataset_preprocess(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(prediction_column)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=3, random_state=0)
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)

        expected = y_test

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        auc_pr = average_precision_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'rf_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))


    ###RF cross validation
    def rf_model_cv(prediction_column):
        print 'you chose RF cross_validation'
        data = dataset_preprocess_cv(prediction_column)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(prediction_column)

        from sklearn.ensemble import RandomForestClassifier
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

        print "RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'rf_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))
  
    ###Naive Bayes
    def nb_model(prediction_column):
        print 'you chose NB'
        data = dataset_preprocess(prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(prediction_column)

        # from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB
        # gnb = GaussianNB()
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

        print "NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc)
        file_name = 'NB_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))


           
    ###Naive Bayes cross validation
    def nb_model_cv(prediction_column):
        print 'you chose NB cross_validation'
        data = dataset_preprocess_cv(prediction_column)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(prediction_column)

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


        print "NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'NB_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))


    def function_chooser(self, A, prediction_column):

        method = getattr(self, A, prediction_column)
        method()
        new_string = 'method called ' + A
        print new_string

    def C(self, prediction_column):
        # train_data = fin
        # prediction_column = fin_2
        # label_type = fin_3
        q = self.function_chooser(self.means, prediction_column)
        return q


def main():

    dataset_file = sys.argv[1]
    model_type = sys.argv[2]


    # dataset = pd.read_csv(dataset_file, sep='\t')
    p = test(means=model_type)
    p.C(dataset_file)



if __name__ == "__main__":
    main()


