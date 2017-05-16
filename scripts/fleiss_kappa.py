# -*- coding: utf-8 -*-
"""
Created on May 15 2017

@author: Alessandro
"""

import pandas as pd


def computeFleissKappa(mat):
    """
    Computes the Fleiss' Kappa value as described in (Fleiss, 1971) 
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Statistics/Fleiss%27_kappa#Python
    """

    n = sum(mat[0])  # PRE : every line count must be equal to n
    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    N = len(mat)
    k = len(mat[0])

    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
        p[j] /= N * n

    # Computing P[]
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))

    # Computing Pbar
    Pbar = sum(P) / N

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj

    kappa = (Pbar - PbarE) / (1 - PbarE)
    return kappa


###T1

file_name = '~/Documents/PhD/relevance_results_new.csv'

file_pd = pd.read_csv(file_name)


grouped = file_pd.groupby('X_unit_id')
ratings = []
fleiss_ratings = []
for group in grouped:
    ratings.append(len(group[1][group[1]['response'] == 'yes']))
    ratings.append(len(group[1][group[1]['response'] == 'no']))
    ratings.append(len(group[1][group[1]['response'] == 'no_item']))
    ratings.append(len(group[1][group[1]['response'] == 'nw_item']))
    ratings.append(len(group[1][group[1]['response'] == 'no_property']))
    ratings.append(len(group[1][group[1]['response'] == 'ne_item']))
    if sum(ratings) == 5:
        fleiss_ratings.append(ratings)
    else:
        print ratings
    ratings = []

T1_fleiss = computeFleissKappa(fleiss_ratings)

###T2
file_name = '~/Documents/PhD/author_all_new.csv'

file_pd = pd.read_csv(file_name)


grouped = file_pd.groupby('ref_value')
ratings = []
fleiss_ratings = []
for group in grouped:
    ratings.append(len(group[1][group[1]['author_type'] == 'organisation']))
    ratings.append(len(group[1][group[1]['author_type'] == 'collective']))
    ratings.append(len(group[1][group[1]['author_type'] == 'nw']))
    ratings.append(len(group[1][group[1]['author_type'] == 'individual']))
    ratings.append(len(group[1][group[1]['author_type'] == 'ne']))
    ratings.append(len(group[1][group[1]['author_type'] == 'dn']))
    if sum(ratings) == 5:
        fleiss_ratings.append(ratings)
    else:
        print ratings
    ratings = []

T2_fleiss = computeFleissKappa(fleiss_ratings)


###T3.A
file_name = '~/Documents/PhD/publisher_all_new.csv'

file_pd = pd.read_csv(file_name)

grouped = file_pd.groupby('domain')
ratings = []
fleiss_ratings = []
for group in grouped:
    ratings.append(len(group[1][group[1]['publisher_type'] == 'news']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'company']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'nw']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'sp_source']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'academia']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'other']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'govt']))
    ratings.append(len(group[1][group[1]['publisher_type'] == 'ne']))
    if sum(ratings) == 5:
        fleiss_ratings.append(ratings)
    else:
        print ratings
    ratings = []

T3A_fleiss  = computeFleissKappa(fleiss_ratings)


###T3.b


file_name = '~/Documents/PhD/publisher_verify_full_new.csv'

file_pd = pd.read_csv(file_name)

grouped = file_pd.groupby('domain')
ratings = []
fleiss_ratings = []
for group in grouped:
    ratings.append(len(group[1][group[1]['results'] == 'vendor']))
    ratings.append(len(group[1][group[1]['results'] == 'no_profit']))
    ratings.append(len(group[1][group[1]['results'] == 'nw']))
    ratings.append(len(group[1][group[1]['results'] == 'cultural']))
    ratings.append(len(group[1][group[1]['results'] == 'political']))
    ratings.append(len(group[1][group[1]['results'] == 'non_trad_news']))
    ratings.append(len(group[1][group[1]['results'] == 'academia_pub']))
    ratings.append(len(group[1][group[1]['results'] == 'trad_news']))
    ratings.append(len(group[1][group[1]['results'] == 'academia_uni']))
    ratings.append(len(group[1][group[1]['results'] == 'academia_other']))
    ratings.append(len(group[1][group[1]['results'] == 'ne']))
    ratings.append(len(group[1][group[1]['results'] == 'no']))
    ratings.append(len(group[1][group[1]['results'] == 'yes']))
    if sum(ratings) == 5:
        fleiss_ratings.append(ratings)
    else:
        print ratings
    ratings = []

T3B_fleiss = computeFleissKappa(fleiss_ratings)
