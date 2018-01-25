# -*- coding: utf-8 -*-
"""
Created on May 1 2017

@author: Alessandro
"""

import os
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import pandas as pd
import scipy as sp
import numpy as np
import chardet
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
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
    train_data = os.path.realpath('../results/prediction_data.csv')
    # train_data = '/Users/alessandro/Documents/WD_references_analysis/results/prediction_data.csv'

    posts = pd.read_csv(train_data, sep='\t', header=0)
    # Create vectorizer for function to use
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    vec = DictVectorizer()
    X_data = sp.sparse.hstack((vectorizer.fit_transform(posts.item_text_clean), vectorizer.fit_transform(posts.object_text), vec.fit_transform(posts[['stat_property', 'user_type', 'code_2',  'instance_of', 'subclass', 'object_instance_of', 'object_subclass', 'property_instance_of']].to_dict(orient='records')).toarray(),posts[['user_edits', 'user_ref_edits_pc', 'ref_count', 'domain_count']].values),format='csr')


    # prediction_column = raw_input('Typeauthoritative or support_object:\n')
    X_train, X_test, y_train, y_test = train_test_split(X_data, posts[prediction_column], test_size=0.3, random_state=53)

    return X_train, X_test, y_train, y_test

def dataset_preprocess_cv(prediction_column):
    train_data = os.path.realpath('../results/prediction_data.csv')
    # train_data = '/Users/alessandro/Documents/WD_references_analysis/results/prediction_data.csv'

    posts = pd.read_csv(train_data, sep='\t', header=0)
    # Create vectorizer for function to use
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    vec = DictVectorizer()

    X_data = sp.sparse.hstack((vectorizer.fit_transform(posts.item_text_clean), vectorizer.fit_transform(posts.object_text), vec.fit_transform(posts[['stat_property', 'stat_value', 'user_type',  'instance_of', 'subclass', 'object_instance_of', 'object_subclass', 'property_instance_of']].to_dict(orient='records')).toarray(),posts[['user_edits', 'user_ref_edits_pc', 'ref_count', 'domain_count']].values),format='csr')
    y = posts[prediction_column]

    return X_data, y


### Model training
class modelTrainer(object):

    def __init__(self):
        values = {'authoritativeness': "authoritative",
                  'relevance': "support_object"}
        models = {'baseline': self.baseline, 'svm_model': self.svm_model, 'svm_model_cv': self.svm_model_cv,
                  'linear_svm': self.linear_svm, 'linear_svm_cv': self.linear_svm_cv,
                  'rf_model': self.rf_model, 'rf_model_cv': self.rf_model_cv,
                  'nb_model': self.nb_model, 'nb_model_cv': self.nb_model_cv}
        self.means = raw_input("Which model do you want to train? "
                               "(Choices: 'baseline', 'svm_model', 'svm_model_cv', 'rf_model', rf_model_cv', 'nb_model', 'nb_model_cv', 'all_cv')\n")
        prediction_col = raw_input("Please choose the attribute you want to evaluate ('authoritativeness' or 'relevance'):")
        self.prediction_column = values[prediction_col]
        if self.means != 'all_cv':
            models[self.means]()
        else:
            self.baseline()
            self.nb_model_cv()
            self.rf_model_cv()
            self.svm_model_cv()

    def baseline(self):
        # train_data = '/Users/alessandro/Documents/WD_references_analysis/results/prediction_data.csv'
        train_data = os.path.realpath('../results/prediction_data.csv')
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
            f.write("Baseline precision:" + str(precision) + "; recall:" + str(recall) + "; f1:" + str(f1) + "; auc_pr:" + str(auc_pr) + "; mcc:" + str(mcc))




    def svm_model(self):
        print 'you chose SVM'
        data = dataset_preprocess(self.prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

        ###SVM
        from sklearn import svm
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
    def svm_model_cv(self):
        print 'you chose SVM cross_validation'
        data = dataset_preprocess_cv(self.prediction_column)
        print 'data processed'

        X_train = data[0]
        y_test = data[1]
        label_type = str(self.prediction_column)

        from sklearn import svm
        from sklearn.model_selection import StratifiedKFold
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

        print "SVM CV model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'svm_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("SVM cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###linear SVM
    def linear_svm(self):
        print 'you chose linear SVM'
        data = dataset_preprocess(self.prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

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

    def linear_svm_cv(self):
        print 'you chose linear SVM cross_validation'
        data = dataset_preprocess_cv(self.prediction_column)
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
                mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(
                mean(mcc_list)) + "; f1_new:" + str(f1_new))

    ###RF
    def rf_model(self):
        print 'you chose RF'
        data = dataset_preprocess(self.prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

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
    def rf_model_cv(self):
        print 'you chose RF cross_validation'
        data = dataset_preprocess_cv(self.prediction_column)
        print 'data processed'

        # X_train = data[0]
        # y_test = data[1]
        label_type = str(self.prediction_column)

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
    def nb_model(self):
        print 'you chose NB'
        data = dataset_preprocess(self.prediction_column)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        label_type = str(self.prediction_column)

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
    def nb_model_cv(self):
        print 'you chose NB cross_validation'
        data = dataset_preprocess_cv(self.prediction_column)
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


        print "NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new)
        file_name = 'NB_cv_results_' + label_type + '.csv'
        with open(file_name, 'w') as f:
            f.write("NB cv model precision:" + str(mean(precision_list)) + "; recall:" + str(mean(recall_list)) + "; f1:" + str(mean(f1_list)) + "; auc_pr:" + str(mean(auc_pr_list)) + "; mcc:" + str(mean(mcc_list)) + "; f1_new:" + str(f1_new))


    # def function_chooser(self):
    #
    #     method = getattr(self.means)
    #     method()
    #     new_string = 'method called ' + self.means
    #     print new_string



def main():

    modelTrainer()



if __name__ == "__main__":
    main()


