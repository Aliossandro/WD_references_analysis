import pandas as pd
import tldextract


eng_tld = ['tv', 'au', 'gov', 'com', 'net', 'org', 'info', 'edu', 'uk', 'edu', 'uk', 'mt', 'eu', 'ca', 'mil',
               'wales', 'nz', 'ph', 'euweb', 'ie', 'id', 'info', 'ac', 'za', 'int', 'london', 'museum']


df_test = pd.read_csv('results/test_w_domain.csv')
df_test['']

url_list_en = [url for url in url_list if tldextract.extract(url).suffix]