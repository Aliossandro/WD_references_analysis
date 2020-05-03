# -*- coding: utf-8 -*-
"""
Created on May 1 2017

@author: Alessandro
"""

import os
import sys

import argparse

import requests

from reference_quality_predictor.data_preprocessing import dataset_preprocess
from reference_quality_predictor.metrics import mean, conf_counter, f1_compute

reload(sys)
sys.setdefaultencoding("utf8")

import pandas as pd

from joblib import dump

from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
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
def save_predicted_data(label_type, predicted, y_test, writing_mode='w'):
    prediction_data = {'predicted': predicted,
                       'expected': y_test}
    prediction_data_df = pd.DataFrame(prediction_data)
    prediction_data_df.to_csv('predicted_{}.csv'.format(label_type), mode=writing_mode)


def extract_webpage_text(link):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
        'Upgrade-Insecure-Requests': '1', 'DNT': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5', 'Accept-Encoding': 'gzip, deflate'}
    try:
        r = requests.get(link, headers=headers, timeout=5, auth=('user', 'pass'))
        return r.text
    except:
        return 'NA'



def check_webpage_support(value_1, value_2, link):
    text = extract_webpage_text(link)
    if is_not_ascii(text):
        text = fix_encode_decode(text)
    if is_not_ascii(value_1):
        value_1 = fix_encode_decode(value_1)
    if is_not_ascii(value_2):
        value_2 = fix_encode_decode(value_2)
    try:
        return 1 if str(value_1.encode('utf-8')) in text and str(value_2.encode('utf-8')) in text else 0
    except UnicodeDecodeError:
        print value_1
        print value_2
        print text
        return 0

def is_not_ascii(string):
    return string is not None and any([ord(s) >= 128 for s in string])

def fix_encode_decode(x):
    try:
        return x.encode('utf-8')
    except UnicodeDecodeError:
        return x.decode('utf-8')


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

        if self.test_data:
            print(self.test_data)
            posts = pd.read_csv(self.test_data)
        else:
            train_data = os.path.realpath('./results/prediction_data.csv')
            posts = pd.read_csv(train_data, sep='\t', header=0)

        if self.prediction_column == 'authoritative':
            try:
                predicted = posts['authoritative_baseline']
            except KeyError:
                authoritative_baseline_file = 'data/authoritative_baseline.csv'
                authoritative_baseline = pd.read_csv(authoritative_baseline_file)
                predicted = posts['authoritative'].apply(lambda x: 0 if x in authoritative_baseline['domain'] else 1)
            expected = posts['authoritative']
        elif self.prediction_column == 'support_object':
            try:
                predicted = posts['statement_match']
            except KeyError:
                predicted = posts.apply(
                    lambda x: check_webpage_support(x['item_labels'], x['stat_value_label'], x['ref_value']), axis=1)

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

        # save model to file
        dump(clf, 'svm_model_{}.joblib'.format(label_type))

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
    def rf_model(self, save_predicted=True):
        print 'you chose RF'
        data = dataset_preprocess(self.prediction_column, self.train_data, self.test_data)
        print 'data processed'

        X_train = data[0]
        y_train = data[2]
        X_test = data[1]
        y_test = data[3]
        data_columns = data[4]
        label_type = str(self.prediction_column)

        clf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=3, random_state=0)
        clf = clf.fit(X_train, y_train)
        feature_importances = pd.DataFrame(clf.feature_importances_,
                                           index=data_columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        feature_importances.to_csv('./rf_feature_importances_{}.csv'.format(label_type))

        predicted = clf.predict(X_test)

        expected = y_test
        if save_predicted:
            save_predicted_data(label_type, predicted, y_test)

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
    def rf_model_cv(self, save_predicted=True):
        print 'you chose RF cross_validation'
        data = dataset_preprocess(self.prediction_column, self.train_data, test_data=None, cv=True)
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
            if save_predicted:
                save_predicted_data(label_type, predicted, y_test, writing_mode='a')

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
