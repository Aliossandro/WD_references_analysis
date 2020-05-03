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


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


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


def f1_compute(tp_list, fp_list, fn_list):
    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)

    f1_score_custom = (2 * tp) / float(2 * tp + fp + fn)

    return f1_score_custom
