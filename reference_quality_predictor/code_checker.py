import json
import sys

import numpy as np
import requests
import grequests
import tldextract


# import resource
# resource.setrlimit(resource.RLIMIT_NOFILE, (11000, 11000))

def get_ref_value_code(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
            'Upgrade-Insecure-Requests': '1', 'DNT': '1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5', 'Accept-Encoding': 'gzip, deflate'}
        r = requests.get(url, headers=headers, timeout=5, auth=('user', 'pass'))
        code = r.status_code
        # code = urllib.request.urlopen(url).code
    except:
        code = 404
    return code


def exception_handler(request, exception):
    return 404


def get_value_code_for_list(url_list):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
        'Upgrade-Insecure-Requests': '1', 'DNT': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5', 'Accept-Encoding': 'gzip, deflate'}
    rs = (grequests.get(u, headers=headers, timeout=5, auth=('user', 'pass')) for u in url_list)
    return grequests.map(rs)
    # return {url: get_ref_value_code(url) for url in url_list}


def load_url_list(file_name):
    url_list = []
    with open(file_name) as f:
        for line in f:
            url_list.append(line.replace('\n', ''))
    return url_list


#
# def applyParallel(split_lists, func):
#     with Pool(cpu_count()) as p:
#         ret_list = p.map(func, [url_list for url_list in split_lists])
#     return list(ret_list)
#
#
def split_list(url_list):
    return [l.tolist() for l in np.array_split(url_list, 1000)]


def extract_codes(file_name):
    url_list = load_url_list(file_name)
    eng_tld = ['tv', 'au', 'gov', 'com', 'net', 'org', 'info', 'edu', 'uk', 'edu', 'uk', 'mt', 'eu', 'ca', 'mil',
               'wales', 'nz', 'ph', 'euweb', 'ie', 'id', 'info', 'ac', 'za', 'int', 'london', 'museum']
    url_list_en = [url for url in url_list if tldextract.extract(url).suffix]
    split_url_list = split_list(url_list)
    extracted_code_list = []
    counter = 0
    for split in split_url_list:
        split_extracted = get_value_code_for_list(split)
        split_extracted = [url.status_code if url is not None else 404 for url in split_extracted]
        print(split_extracted[:15])
        extracted_code_list.append(dict(zip(split, split_extracted)))
        counter += 0
        print(counter)
    extracted_code_dict = {k: v for d in extracted_code_list for k, v in d.items()}
    return extracted_code_dict


def main():
    file_path = sys.argv[1]
    df = extract_codes(file_path)
    with open('url_code_dictionary.json', 'w') as f:
        json.dump(df, f)


if __name__ == "__main__":
    main()
