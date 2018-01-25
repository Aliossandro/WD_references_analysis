# WD external references analysis
Code and datasets extracted for the analysis of Wikidata external references

### Python and sklearn versions
The experiment was carried out using Python 2.7.13 and sklearn 0.18.1

### Algorithms' parameters
We tried different parameters for each of the algorithms used. We chose those that best adapted to our dataset characteristics or that yielded the best performance.

# Naive Bayes
Naive Bayes learners can make different assumptions regarding the distribution of the data.
In this work we used a Naive Bayes classifier based on a Bernoulli distribution.

# Random Forest
Random Forest allows users to tune several parameters. We did not notice significant improvements by testing different settings.
These are the parameters set:
Number of trees in the forest ('n_estimators') = 1000
Number of features considered when looking for the best split (max_features) = default (square root of total number of features)
Maximum depth of the tree (max_depth) = None (i.e. nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples - from http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit)
Minimum number of instances required to split a node (min_samples) = 3 

# SVM
We tested two different kernels: 'linear' and 'rbf'.
Rbf offered the best performance. The results provided in the paper refer to this kernel.
'C' sets the penalty for misclassified instances. Its values are between 0 and 1, we set it to .4 (for rbf).
Cache size was set to 1000 (in MB).
The 'class_weight' option was set to 'balanced'. This means that weights were adjusted in an inversely proportional relation to class frequencies of output variables (n_samples / (n_classes * np.bincount(y)) (from http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
