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


# posts = pd.read_csv('./final_model_refs.csv', sep = '\t')
#
# feature_ref = [
# {'stat_property': 'P136', 'stat_item': 'Q5069525', 'item_id': 'Q221358', 'author_type': 'human', 'user_edits': 107314, 'user_ref_edits': 2592, 'item_text': parso_altro},
# {'stat_property': 'P136', 'stat_item': 'Q211756', 'item_id': 'Q1165439', 'author_type': 'human', 'user_edits': 201516, 'user_ref_edits': 25027, 'item_text': parsoco},
# {'stat_property': 'P136', 'stat_item': 'Q1770695', 'item_id': 'Q981797', 'author_type': 'human', 'user_edits': 71314, 'user_ref_edits': 1201, 'item_text': parsed_sente}]
#
# vec = DictVectorizer()
# vec.fit_transform(feature_ref).toarray()
#
# vec.get_feature_names()
#
#
# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
#


####starts HERE
#
# authority_file = '~/Documents/PhD/WD_refs_results/authoritative_all.csv'
# text_file = '~/Documents/PhD/WD_refs_results/revision_refs_clean_new.csv'
#
# text_pd = pd.read_csv(text_file, sep='\t', header=0)
# authority_all = pd.read_csv(authority_file)

# # df = pd.read_csv(text_file, sep='\t', iterator=True, chunksize=50000)
# # for chunk in df:
# #
# #     chunk['time_stamp'] = pd.to_datetime(chunk['time_stamp'])
# #     chunk = chunk.groupby(['item_id'])['time_stamp'].transform(max) == df['time_stamp']
# #     print chunk.head()
# #
# #
# # inds = df.groupby(['Author'])['Val'].transform(max) == df['Val']
# # df = df[inds]
# # df.reset_index(drop=True, inplace=True)



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
        train_data = './prediction_data.csv'  # './prediction_no_census.csv'
        posts = pd.read_csv(train_data, sep='\t', header=0)

        prediction_column = raw_input("authority_baseline or statement_match?\n")

        if prediction_column == 'authority_baseline':
            expected_column = 'authoritative'
        elif prediction_column == 'statement_match':
            expected_column = 'support_object'

        predicted = posts[prediction_column]
        expected = posts[expected_column]

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        roc = roc_auc_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc)
        file_name = 'baseline_results_' + str(prediction_column) + '.csv'
        with open(file_name, 'w') as f:
            f.write("Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc))




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
        roc = roc_auc_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc)
        file_name = 'svm_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc))

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
        roc_list = []
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
            roc = roc_auc_score(expected, predicted, average='weighted')
            roc_list.append(roc)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)

        # clf = clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test)

        # expected = y_test
        #
        # precision = precision_score(expected, predicted, average='weighted', pos_label=0)
        # recall = recall_score(expected, predicted, average='weighted', pos_label=0)
        # f1 = f1_score(expected, predicted, average='weighted', pos_label=0)
        # roc = roc_auc_score(expected, predicted, average='weighted')
        # mcc = matthews_corrcoef(expected, predicted)

        print "SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))

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
        roc = roc_auc_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "Linear SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
            f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc)
        file_name = 'linear_svm_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("Linear SVM model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(
                f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc))

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

        # crossvalidation = KFold(n_splits=10, shuffle=True, random_state=3)
        # score_type = raw_input('precision, recall, roc_auc, f1, or matthews_corrcoef:\n')
        # scores = cross_val_score(clf, X_train, y_test, scoring=score_type, cv=crossvalidation, n_jobs=4)
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        f1_list = []
        roc_list = []
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
            roc = roc_auc_score(expected, predicted, average='weighted')
            roc_list.append(roc)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)

        # clf = clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test)

        # expected = y_test
        #
        # precision = precision_score(expected, predicted, average='weighted', pos_label=0)
        # recall = recall_score(expected, predicted, average='weighted', pos_label=0)
        # f1 = f1_score(expected, predicted, average='weighted', pos_label=0)
        # roc = roc_auc_score(expected, predicted, average='weighted')
        # mcc = matthews_corrcoef(expected, predicted)

        print "Linear SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(
            mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(
            mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'linear_svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("Linear SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(
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
        clf = RandomForestClassifier(n_estimators=4, max_depth=None, min_samples_split=3, random_state=0)
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)

        expected = y_test

        precision = precision_score(expected, predicted, average='weighted', pos_label=1)
        recall = recall_score(expected, predicted, average='weighted', pos_label=1)
        f1 = f1_score(expected, predicted, average='weighted', pos_label=1)
        roc = roc_auc_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc)
        file_name = 'rf_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc))


    ###RF cross validation
    def rf_model_cv(prediction_column):
        print 'you chose RF cross_validation'
        data = dataset_preprocess_cv(prediction_column)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(prediction_column)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=4, max_depth=None, min_samples_split=2, random_state=0)
        crossvalidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        precision_list = []
        recall_list = []
        roc_list = []
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
            roc = roc_auc_score(expected, predicted, average='weighted')
            roc_list.append(roc)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)


        # clf = clf.fit(X_train, y_train)
        # predicted = clf.predict(X_test)

        # expected = y_test
        #
        # precision = precision_score(expected, predicted, average='weighted', pos_label=0)
        # recall = recall_score(expected, predicted, average='weighted', pos_label=0)
        # f1 = f1_score(expected, predicted, average='weighted', pos_label=0)
        # roc = roc_auc_score(expected, predicted, average='weighted')
        # mcc = matthews_corrcoef(expected, predicted)

        print "RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'rf_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("RF cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))
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
        roc = roc_auc_score(expected, predicted, average='weighted')
        mcc = matthews_corrcoef(expected, predicted)

        print "NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc)
        file_name = 'NB_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB model precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; roc:" + str(roc) + "; mcc:" + str(mcc))


            ###RF cross validation

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
        roc_list = []
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
            roc = roc_auc_score(expected, predicted, average='weighted')
            roc_list.append(roc)
            mcc = matthews_corrcoef(expected, predicted)
            mcc_list.append(mcc)

        f1_new = f1_compute(true_positive_list, false_positive_list, false_negative_list)


        print "NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'NB_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; roc:" + str(mean(roc_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))


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
    # pred_column = sys.argv[2]
    model_type = sys.argv[2]


    # dataset = pd.read_csv(dataset_file, sep='\t')
    p = test(means=model_type)
    p.C(dataset_file)



if __name__ == "__main__":
    main()

### Authoritativeness prediction
###baseline
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
#
# path = '/Users/alessandro/Documents/PhD/WD_refs_results/baseline.csv'
#
# baseline = pd.read_csv(path)
#
#
#
#
# ###split train test
# ###decode text
# lb = preprocessing.LabelBinarizer()
# y_data_b = lb.fit_transform(posts.authoritative)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_b, test_size=0.3, random_state=12)
#
# ###SVM
# from sklearn import svm
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#
#
# ###compute scores
# clf.score(X_test, y_test)
# predicted = clf.predict(X_test)
# expected = y_test
#
#
# precision_score(expected, predicted, average=None)
# recall_score(expected, predicted, average=None)
# f1_score(expected, predicted, average=None)
# roc_auc_score(expected, predicted, average=None)
#
#
# ###RF
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=4, max_depth=None, min_samples_split=2, random_state=0, oob_score=True)
# clf = clf.fit(X_train, y_train)
# predicted = clf.predict(X_test)
#
# precision_score(expected, predicted, average=None)
# recall_score(expected, predicted, average=None)
# f1_score(expected, predicted, average=None)
# roc_auc_score(expected, predicted, average=None)
#
# #
# #
# # clf = svm.SVC(probability=True, random_state=0)
# # cross_val_score(clf, X_train, y_train, scoring='precision')
#
#
# ###Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import BernoulliNB
# gnb = GaussianNB()
# ber = BernoulliNB()
# y_pred = gnb.fit(X_train.toarray(), y_train).predict(X_test.toarray())
# y_pred = ber.fit(X_train.toarray(), y_train).predict(X_test.toarray())
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
#
# precision_score(expected, predicted, average=None)
# recall_score(expected, predicted, average=None)
# f1_score(expected, predicted, average=None)
# roc_auc_score(expected, predicted, average=None)
#
#
#
#
#
#
#
#
#
# ###cross-validation
#
# scores = cross_val_score(clf, X_train, y_train, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# crossvalidation = KFold(n_splits=10,  shuffle=True, random_state=3)
# scores = cross_val_score(clf, X_data, posts.support_object, scoring=('precision', 'recall', 'roc_auc', 'f1') , cv=crossvalidation,  n_jobs=4)
#
#
# print ‘Folds: %i, mean squared error: %.2f std: %.2f’
#  %(len(scores),np.mean(np.abs(scores)),np.std(scores))
# Folds: 10, mean squared error: 23.76 std: 12.13
#
#
#
# kf = KFold(n_splits=10)
# for train, test in kf.split(X_data):
#     print("%s %s" % (train, test))
#
#
#
# X_train_data = vec.fit_transform(X_train[['ref_value', 'ref_domain', 'stat_property', 'stat_value', 'item_id', 'user_type', 'user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].to_dict(orient = 'records')).toarray()
# X_test_data = vec.fit_transform(X_test[['ref_value', 'ref_domain', 'stat_property', 'stat_value', 'item_id', 'user_type', 'user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].to_dict(orient = 'records')).toarray()
#
#
#
# X = sp.sparse.hstack((vec.fit_transform(posts[['ref_value', 'ref_domain', 'stat_property', 'stat_value', 'item_id', 'user_type']].to_dict(orient = 'records')).toarray(), posts[['user_edits', 'user_ref_edits', 'ref_count', 'domain_count']].values),format='csr')
#
#
# X = sp.sparse.hstack((vectorizer.fit_transform(posts.item_text), vec.fit_transform(posts[['stat_property','stat_item', 'item_id', 'author_type']].to_dict(orient = 'records')).toarray(), posts[['user_edits', 'user_ref_edits']].values),format='csr')
#
#
#
#
#
#
#
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# clf = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=2, random_state=0, oob_score=True)
# clf = clf.fit(data[0], data[1])
# y_pred = clf.predict(data[0])
# print("Number of mislabeled points out of a total %d points : %d" % (data[0.shape[0],(posts.support_object != y_pred).sum()))
#
#
# importances = clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# counter = 0
# for f in range(data[0].shape[1]):
#     counter += 1
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#     if counter == 100:
#         break
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(data[0].shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(data[0].shape[1]), indices)
# plt.xlim([-1, data[0].shape[1]])
# plt.show()
#
#
#
# scores = cross_val_score(clf, X_train_data, X_train.support_object)
#
#
#
#
#
#
#
#
#
#
# #
# # from sklearn import datasets
# # iris = datasets.load_iris()
# # from sklearn.naive_bayes import GaussianNB
# # gnb = GaussianNB()
# # y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# #  print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum())) Number of mislabeled points out of a total 150 points : 6
#
#
# import matplotlib.pyplot as plt
#
# from collections import OrderedDict
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#
# # Author: Kian Ho <hui.kian.ho@gmail.com>
# #         Gilles Louppe <g.louppe@gmail.com>
# #         Andreas Mueller <amueller@ais.uni-bonn.de>
# #
# # License: BSD 3 Clause
#
#
#
#
#
# print(__doc__)
#
# RANDOM_STATE = 12
#
# # Generate a binary classification dataset.
# X, y = make_classification(n_samples=500, n_features=25,
#                            n_clusters_per_class=1, n_informative=15,
#                            random_state=RANDOM_STATE)
#
# # NOTE: Setting the `warm_start` construction parameter to `True` disables
# # support for parallelized ensembles but is necessary for tracking the OOB
# # error trajectory during training.
# ensemble_clfs = [
#     ("RandomForestClassifier, max_features='sqrt'",
#         RandomForestClassifier(warm_start=True, oob_score=True,
#                                max_features="sqrt",
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features='log2'",
#         RandomForestClassifier(warm_start=True, max_features='log2',
#                                oob_score=True,
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features=None",
#         RandomForestClassifier(warm_start=True, max_features=None,
#                                oob_score=True,
#                                random_state=RANDOM_STATE))
# ]
#
# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
#
# # Range of `n_estimators` values to explore.
# min_estimators = 3
# # max_estimators = 10
#
# for label, clf in ensemble_clfs:
#     for i in range(min_estimators, max_estimators + 1):
#         clf.set_params(n_estimators=4)
#         clf.fit(data[0], data[1])
#
#         # Record the OOB error for each `n_estimators=i` setting.
#         oob_error = 1 - clf.oob_score_
#         error_rate = oob_error
#
# # Generate the "OOB error rate" vs. "n_estimators" plot.
# for label, clf_err in error_rate.items():
#     xs, ys = zip(*clf_err)
#     plt.plot(xs, ys, label=label)
#
# plt.xlim(min_estimators, max_estimators)
# plt.xlabel("n_estimators")
# plt.ylabel("OOB error rate")
# plt.legend(loc="upper right")
# plt.show()
#
