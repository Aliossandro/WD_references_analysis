import pandas as pd
import ujson
import re
import chardet


def claim_maker(line):
    try:
        strc = eval(line)
        print strc
        claimi = strc['claims']
        return claimi
    except ValueError as e:
        print e
        print line


authority_file = '~/Documents/PhD/WD_refs_results/authoritative_all_new.csv'
text_file = '~/Documents/PhD/WD_refs_results/revision_refs_clean_new.csv'

text_pd = pd.read_csv(text_file, sep='\t', header=0)

# prop_text_file = '~/Documents/PhD/WD_refs_results/all_items_clean.csv'
#
# prop_text_pd = pd.read_csv(prop_text_file, sep='\t', header=0)


# prop_text_pd['claims_text'] = prop_text_pd['claims'].map(lambda x : claim_maker(x))

authority_all = pd.read_csv(authority_file)

text_pd = text_pd[['claims', 'item_id']]

file_name = '~/Documents/PhD/WD_refs_results/refs_count.csv'

posts = pd.read_csv(file_name)

file_name_user = '~/Documents/PhD/WD_refs_results/user_edits_refs.csv'

user_data = pd.read_csv(file_name_user)

posts = posts.merge(user_data, how='left', on='rev_id')
posts = posts.merge(text_pd, how='right', on='item_id')

posts = posts[['rev_id', 'ref_value', 'ref_domain', 'ref_count', 'domain_count', 'stat_property', 'stat_value', 'item_id', 'user_type', 'user_edits', 'user_ref_edits', 'claims']]

relevance_data = '~/Documents/PhD/WD_refs_results/relevance_total_support.csv'

relevance_data = pd.read_csv(relevance_data)

relevance_data = relevance_data[['rev_id', 'support_object', 'X_unit_id']]


posts = posts.merge(relevance_data, on=['rev_id'])

ref_codes_file = '~/Documents/PhD/WD_refs_results/ref_codes_final.csv'
ref_codes = pd.read_csv(ref_codes_file)

posts = posts.merge(ref_codes, on=['rev_id'])

posts.support_object[posts['support_object'] == 'Page not working'] = 'No'
posts = posts[posts['support_object'] != 'Page not in English']
posts = posts.merge(authority_all, on=['rev_id', 'ref_value'])

baseline_data = './page_matches.csv'
baseline_statement = pd.read_csv(baseline_data)
baseline_statement = baseline_statement[['rev_id', 'ref_value', 'item_match', 'object_match', 'statement_match']]
posts = posts.merge(baseline_statement, on=['rev_id', 'ref_value'])
posts['user_ref_edits_pc'] = posts['user_ref_edits']/posts['user_edits']

posts = posts.drop_duplicates('X_unit_id')

# posts = pd.DataFrame(feature_ref)
###decode text
posts['item_text'] = posts['claims'].map(lambda x: x.decode(chardet.detect(x)['encoding']))
# lb = preprocessing.LabelBinarizer()
# y_data_b = lb.fit_transform(posts.support_object)


posts.support_object[posts['support_object'] == 'Yes'] = 0
posts.support_object[posts['support_object'] == 'No'] = 1
# posts.support_object = posts.support_object.astype(int)

posts.authoritative = posts.authoritative.astype(str)
posts.authoritative[posts['authoritative'] == 'True'] = 0
posts.authoritative[posts['authoritative'] == 'False'] = 1

posts = posts.drop('claims', axis = 1)
posts = posts.drop('X_unit_id', axis = 1)
posts.user_type[posts['user_type'] == 'anonymous'] = 0
posts.user_type[posts['user_type'] == 'human'] = 1
posts.user_type[posts['user_type'] == 'bot'] = 2
posts.code_2[posts['code_2'] == 'not working'] = 999

word_list = ["u'type", "u'statement", "u'references", "u'snaks", "u'datavalue", "u'type", "u'value", "u'entity-type", "u'property", "u'snaktype", "u'mainsnak", "u'rank", "u'"]
word_remove = '|'.join(word_list)
pattern = re.compile(word_remove)
posts['item_text_clean'] = posts['item_text'].map(lambda x: pattern.sub(' ', x))
pattern = re.compile('[\W_]+')
posts['item_text_clean'] = posts['item_text_clean'].map(lambda x: pattern.sub(' ', x))

wd_hierarchy_file = '~/Documents/PhD/WD_refs_results/wd_hierarchy_clean.csv'
wd_hierarchy = pd.read_csv(wd_hierarchy_file)
wd_hierarchy = wd_hierarchy[['item_id', 'instance_of', 'subclass']]

posts = posts.merge(wd_hierarchy, how = 'left', on='item_id')

wd_hierarchy_file = '~/Documents/PhD/WD_refs_results/object_hierarchy_clean.csv'
wd_hierarchy = pd.read_csv(wd_hierarchy_file)
wd_hierarchy = wd_hierarchy[['item_id', 'instance_of', 'subclass']]
wd_hierarchy.columns = ['item_id', 'object_instance_of', 'object_subclass']

posts = posts.merge(wd_hierarchy, how = 'left', on='item_id')

wd_hierarchy_file = '~/Documents/PhD/WD_refs_results/property_hierarchy_clean.csv'
wd_hierarchy = pd.read_csv(wd_hierarchy_file)
wd_hierarchy = wd_hierarchy[['stat_property', 'subproperty']]
wd_hierarchy.columns = ['stat_property', 'property_instance_of']

posts = posts.merge(wd_hierarchy, how = 'left', on='stat_property')

def dec(line):
    try:
        line = line.decode(chardet.detect(line)['encoding'])
        return line
    except ValueError:
        return line

all_items_file = './all_items_clean.csv'
all_items = pd.read_csv(all_items_file, sep = '\t')
all_items = all_items[['item_id', 'claims']]
all_items.columns = ['stat_value', 'object_text']
all_items['object_text'] = all_items['object_text'].map(lambda x: dec(x))
word_list = ["u'type", "u'statement", "u'references", "u'snaks", "u'datavalue", "u'type", "u'value", "u'entity-type", "u'property", "u'snaktype", "u'mainsnak", "u'rank", "u'"]
word_remove = '|'.join(word_list)
pattern = re.compile(word_remove)
all_items['object_text'] = all_items['object_text'].map(lambda x: pattern.sub(' ', x))
pattern = re.compile('[\W_]+')
all_items['object_text'] = all_items['object_text'].map(lambda x: pattern.sub(' ', x))


posts = posts.merge(all_items, how = 'left', on='stat_value')
posts.object_text[posts['object_text'].isnull()] = posts.stat_value[posts['object_text'].isnull()]


posts['ref_domain'] = posts['ref_domain'].map(lambda x : x.replace('www.', ''))


baseline_data = './authoritative_baseline.csv'
baseline = pd.read_csv(baseline_data)

posts['authority_baseline'] = 0
posts['authority_baseline'][posts['ref_domain'].isin(baseline['domain'])] = 1

posts['statement_match'] = 1
posts['statement_match'][(posts['item_match'] > 0) & (posts['object_match'] > 0)] = 0

posts = posts.fillna(value=0)