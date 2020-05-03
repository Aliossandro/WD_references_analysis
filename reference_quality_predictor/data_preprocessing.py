import pandas as pd
import scipy as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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
    item_vec = item_vectorizer.fit_transform(train_df['item_data'])
    item_vocab = item_vectorizer.vocabulary_.items()
    object_vec = object_vectorizer.fit_transform(train_df['object_data'])
    object_vocab = object_vectorizer.vocabulary_.items()
    vec_features = vec.fit_transform(train_df[['stat_property',
                                               'user_type',
                                               'http_code']].to_dict(orient='records')).toarray()
    X_data = sp.sparse.hstack((item_vec,
                               object_vec, vec_features,
                               train_df[['user_edits', 'user_ref_edits_p', 'url_use', 'domain_use']].values),
                              format='csr')
    print(X_data.shape)
    rev_dictionary = {v: k for k, v in item_vocab}
    column_names_from_item_features = [v for k, v in rev_dictionary.items()]
    rev_dictionary = {v: k for k, v in object_vocab}
    column_names_from_object_features = [v for k, v in rev_dictionary.items()]

    data_columns = column_names_from_item_features + column_names_from_object_features + vec.feature_names_ + [
        'user_edits', 'user_ref_edits_p',
        'url_use', 'domain_use']

    if cv:
        y = train_df[prediction_column]

        return X_data, y, data_columns
    else:
        # prediction_column = raw_input('Type authoritative or support_object:\n')
        if test_data:
            test_df = pd.read_csv(test_data)
            item_vec_test = item_vectorizer.transform(test_df['item_data'])
            object_vec_test = object_vectorizer.transform(test_df['object_data'])
            vec_features_test = vec.transform(test_df[['stat_property',
                                                       'user_type',
                                                       'http_code']].to_dict(orient='records')).toarray()
            X_test = sp.sparse.hstack((item_vec_test,
                                       object_vec_test, vec_features_test,
                                       test_df[['user_edits', 'user_ref_edits_p', 'url_use', 'domain_use']].values),
                                      format='csr')
            print(X_test.shape)

            y_train = train_df[prediction_column]
            y_test = test_df[prediction_column]

            return X_data, X_test, y_train, y_test, data_columns

        X_train, X_test, y_train, y_test = train_test_split(X_data, train_df[prediction_column],
                                                            test_size=0.3,
                                                            random_state=53)

        return X_train, X_test, y_train, y_test, data_columns
