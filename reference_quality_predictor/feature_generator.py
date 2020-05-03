import json
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import psycopg2

# # import pickle


# # counterS = 0
# # global counterS
# # global valGlob
# # from sqlalchemy import create_engine
#
# # -*- coding: utf-8 -*-
# import os
# import sys
# import copy


# fileName = '/Users/alessandro/Documents/PhD/OntoHistory/WDTaxo_October2014.csv'


# connection parameters
import requests


def get_db_params():
    params = {
        'database': 'wikidb',
        'user': 'postgres',
        'password': 'postSonny175',
        'host': 'localhost',
        'port': '5432'
    }
    conn = psycopg2.connect(**params)
    return conn


def print_query(query):
    print(f'Now running this query: {query}')


def applyParallel(dfGrouped, func, user_grouped):
    with Pool(cpu_count()) as p:
        df_func = lambda x: func(user_grouped, x)
        ret_list = p.map(df_func, [(name, group) for name, group in dfGrouped])
    return pd.concat(ret_list)


def get_no_edits_parallel(df, group_user_tuple):
    user = group_user_tuple[0]
    group = group_user_tuple[1]
    temp_df = df.get_group(user)
    group['user_edits'] = group['time_stamp'].apply(
        lambda x: get_no_edits(temp_df, x))
    return group


def get_no_edits(df, date):
    return df.loc[df['time_stamp'] <= date,].shape[0]


def generate_user_query(items):
    query = """SELECT user_name, time_stamp FROM user_edits_feature
    WHERE user_name IN (
    SELECT users_needed FROM (VALUES """ + items + """)
    t3 (users_needed) );"""

    return query


def get_user_data_sample(conn, items):
    query = generate_user_query(items)
    print_query(query)
    df = pd.DataFrame()
    for chunk in pd.read_sql(query, con=conn, chunksize=100000):
        df = df.append(chunk)
    print(f'User dataframe size: {df.shape[0]}')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df


def get_user_data(conn):
    query = """SELECT user_name, time_stamp FROM user_edits_feature;"""
    print_query(query)
    df = pd.DataFrame()
    for chunk in pd.read_sql(query, con=conn, chunksize=100000):
        df = df.append(chunk)
    print(f'User dataframe size: {df.shape[0]}')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df


def get_refs_edits(conn, user_name):
    query_count = """SELECT COUNT(*) FROM reference_clean_new WHERE time_stamp < '2016-10-01' 
    AND user_name = '""" + user_name + """';
    """
    cur = conn.cursor()
    print_query(query_count)
    cur.execute(query_count)
    results = cur.fetchone()
    cur.close()
    conn.commit()
    # refs_edits = df.loc[df['user_name'] == user_name,].shape[0]
    return results['count']


def generate_item_query(items):
    query = """SELECT itemid AS item_id, statproperty AS stat_property, statvalue AS stat_value FROM statementsdata_201710 
    WHERE itemid IN (
    SELECT items_needed FROM (VALUES """ + items + """)
    t3 (items_needed) ) AND statvalue != 'deleted' AND statvalue ~* '^[Q][0-9]{1,}';"""
    return query


def get_item_data(conn, items):
    query = generate_item_query(items)
    # print(query)
    df = pd.DataFrame()
    for chunk in pd.read_sql(query, con=conn, chunksize=100000):
        df = df.append(chunk)
    print(f'Dataframe size: {df.shape[0]}')
    return df


def generate_property_query(items):
    query = """SELECT item_id, stat_property, stat_value FROM property_statements 
    WHERE item_id IN (
    SELECT items_needed FROM (VALUES """ + items + """)
    t3 (items_needed) ) AND stat_value != 'deleted' AND stat_value ~* '^[Q][0-9]{1,}';"""
    return query


def get_property_data(conn, items):
    query = generate_property_query(items)
    # print(query)
    df = pd.DataFrame()
    for chunk in pd.read_sql(query, con=conn, chunksize=1000):
        df = df.append(chunk)
    print(f'Dataframe size: {df.shape[0]}')
    return df


def get_item_values(df):
    property_values = df['stat_property'].tolist()
    object_values = df['stat_value'].tolist()
    item_values = property_values + object_values
    return ' '.join(item_values)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_ref_value_code(url):
    try:
        r = requests.get(url, auth=('user', 'pass'))
        code = r.status_code
    except:
        code = 404
    return code


def get_features(bot_file, refs_file=None):
    conn = get_db_params()

    # get refs data
    if refs_file:
        refs_df = pd.read_csv(refs_file)
    else:
        query_refs = """WITH temp_refs AS (SELECT rev_id, reference_id, user_name, ref_property, ref_value, stat_property, 
        stat_value, item_id, ref_domain 
        FROM reference_enriched)
        SELECT r.rev_id, r.reference_id, r.user_name, r.ref_property, r.ref_value, r.stat_property, r.stat_value, r.item_id, 
        r.ref_domain, s.time_stamp
        FROM temp_refs r, revision_history_201710 s
        WHERE r.rev_id = s.rev_id;"""
        print_query(query_refs)
        refs_df = pd.DataFrame()
        for chunk in pd.read_sql(query_refs, con=conn, chunksize=100000):
            refs_df = refs_df.append(chunk)
    refs_df['time_stamp'] = pd.to_datetime(refs_df['time_stamp'])
    refs_df['date'] = refs_df['time_stamp'].apply(lambda x: x.date().strftime('%Y-%m-%d'))
    print('ref dataset loaded')

    # # get item and object data
    def retrieve_item_text(key, item_dict):
        try:
            text = item_dict[key]
            return text
        except KeyError:
            return type(key)

    unique_items = list(set(refs_df['item_id'].unique().tolist() + refs_df.loc[
        refs_df['stat_value'].str.match(r'[Q][0-9]{1,}'), 'stat_value'].unique().tolist()))
    print(len(unique_items))
    unique_items = [l.tolist() for l in np.array_split(unique_items, 10)]
    item_df_list = []
    for item_list in unique_items:
        print(len(item_list))
        item_values = "('" + "'), ('".join(item_list) + "')"
        item_df_list.append(get_item_data(conn, item_values))
    item_df = pd.concat(item_df_list)
    item_df_grouped = item_df.groupby('item_id')
    refs_items_dict = {df_item[0]: get_item_values(df_item[1]) for df_item in item_df_grouped}

    refs_df['item_data'] = refs_df['item_id'].apply(lambda x: retrieve_item_text(x, refs_items_dict))
    print('item data collected')

    refs_df['object_data'] = refs_df['stat_value'].apply(lambda x: retrieve_item_text(x, refs_items_dict))
    print('object data collected')

    #
    def retrieve_url_no(key, url_dict):
        try:
            text = url_dict[key]
            return text
        except KeyError:
            return key

    #
    # # get property data
    # try:
    unique_properties = refs_df['stat_property'].unique().tolist()
    print(len(unique_properties))
    unique_properties = [l.tolist() for l in np.array_split(unique_properties, 5)]
    property_df_list = []
    for property_list in unique_properties:
        print(len(property_list))
        property_values = "('" + "'), ('".join(property_list) + "')"
        property_df_list.append(get_property_data(conn, property_values))
    property_df = pd.concat(property_df_list)
    property_df_grouped = property_df.groupby('item_id')
    refs_property_dict = {df_property[0]: get_item_values(df_property[1]) for df_property in property_df_grouped}

    refs_df['property_data'] = refs_df['stat_property'].apply(lambda x: retrieve_url_no(x, refs_property_dict))
    print('property data collected')
    # except:
    #     refs_df.to_csv('reference_features_items.csv', index=False)
    #
    # # get domain usage
    # # get item and object data

    try:
        domain_use = refs_df['ref_domain'].value_counts()
        refs_df['domain_use'] = refs_df['ref_domain'].apply(lambda x: retrieve_url_no(x, domain_use))
        print('domain data collected')
    except:
        print('domain data not collected')
        refs_df.to_csv('reference_features_ref_edits.csv', index=False)

    # # get URL usage
    try:
        url_use = refs_df['ref_value'].value_counts()
        refs_df['url_use'] = refs_df['ref_value'].apply(lambda x: retrieve_url_no(x, url_use))
        print('url data collected')
    except:
        print('url data not collected')
        refs_df.to_csv('reference_features_domains.csv', index=False)

    # get user data
    # try:
    unique_users = refs_df['user_name'].unique().tolist()
    unique_users = [user.replace("'", "\\'") for user in unique_users]
    user_values = "(E'" + "'), (E'".join(unique_users) + "')"

    # print(len(unique_users))
    # unique_users = [l.tolist() for l in np.array_split(unique_users, 20)]
    def get_user_edits(key1, key2, user_dict):
        try:
            edit_no = user_dict[key1][key2]
            return edit_no
        except KeyError:
            return 0

    user_df = get_user_data_sample(conn, user_values)
    user_df.sort_values(by=['user_name', 'time_stamp'], inplace=True)
    # user_df.set_index('time_stamp', inplace=True)
    user_df_grouped = user_df.groupby('user_name')
    print('user data dowloaded')
    refs_df_grouped = refs_df.groupby('user_name')
    # refs_df = applyParallel(refs_df_grouped, get_no_edits_parallel, user_df_grouped)
    user_edits_dict = {}
    for user, refs_group in refs_df_grouped:
        date_list = refs_group['date'].unique().tolist()
        try:
            temp_df = user_df_grouped.get_group(user)
            date_dict = {date: get_no_edits(temp_df, date) for date in date_list}
        except KeyError:
            date_dict = {date: 0 for date in date_list}

        user_edits_dict[user] = date_dict

    #     # refs_group['user_edits'] = refs_group['time_stamp'].apply(
    #     #     lambda x: get_no_edits(temp_df, x))
    try:
        refs_df['user_edits'] = refs_df[['user_name', 'date']].apply(
            lambda x: get_user_edits(x['user_name'], x['date'], user_edits_dict), axis=1)
        # refs_df = pd.concat([group for name, group in refs_df_grouped])
        print('user data collected')
    except:
        with open('user_dict_file.json', 'w') as f:
            json.dump(user_edits_dict, f)

    # except:
    #     print('user data not collected')
    #     refs_df.to_csv('reference_features_urls.csv', index=False)

    # # get refs edits
    def retrieve_edit_no(key, date, url_dict):
        try:
            edit_no = url_dict[key][date]
            return edit_no
        except KeyError:
            return 0

    # # unique_users = refs_df['user_name'].unique()
    try:
        user_count = refs_df[['user_name', 'date']].size()
        # refs_edits_dict = {u: get_refs_edits(conn, u) for u in unique_users}
        # refs_df['user_ref_edits'] = refs_df['user_name'].apply(lambda x: retrieve_edit_no(x, user_count))
        refs_df['user_ref_edits'] = refs_df[['user_name', 'date']].apply(
            lambda x: retrieve_edit_no(x['user_name'], x['date'], user_count), axis=1)
        refs_df['user_ref_edits'] = refs_df['user_ref_edits'] / refs_df['user_edits']
        refs_df.fillna(value=0, inplace=True)
        print('user edit data collected')
    except:
        print('user edit data not collected')
        refs_df.to_csv('reference_features_user_edits.csv', index=False)
    #
    # # user type
    bot_list = pd.read_csv(bot_file)
    refs_df['user_type'] = 0
    refs_df.loc[refs_df['user_name'].isin(bot_list),]['user_type'] = 1
    refs_df.loc[refs_df['user_name'].str.contains(
        '([0-9]{1,3}[.]){3}[0-9]{1,3}|(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])[.]){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))',
        regex=True),]['user_type'] = 2

    # get url codes
    refs_df['http_code'] = refs_df['ref_value'].apply(lambda x: get_ref_value_code(x))

    return refs_df


def main():
    bot_file_path = sys.argv[1]
    file_path = sys.argv[2]
    df = get_features(bot_file_path, file_path)
    df.to_csv('reference_features_sample.csv', index=False)


if __name__ == "__main__":
    main()
