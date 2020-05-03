import re
import requests
import json


def get_item_label(item):
    if re.match(r'[Q|P][0-9]{1,}', item):
        url = 'https://www.wikidata.org/wiki/Special:EntityData/' + item + '.json'
        r = requests.get(url, auth=('user', 'pass'))

        try:
            json_data = r.json()
            try:
                json_data['entities'][item]
            except KeyError:
                item = list(json_data['entities'].keys())[0]
            except json.decoder.JSONDecodeError:
                pass
            try:
                label = json_data['entities'][item]['labels']['en']['value']
            except KeyError:
                try:
                    label = json_data['entities'][item]['labels']['de']['value']
                except KeyError:
                    try:
                        label = json_data['entities'][item]['labels']['es']['value']
                    except KeyError:
                        try:
                            label = json_data['entities'][item]['labels']['fr']['value']
                        except KeyError:
                            try:
                                label = json_data['entities'][item]['labels']['nl']['value']
                            except KeyError:
                                label = json_data['entities'][item]['labels']
            except json.decoder.JSONDecodeError:
                return item
            return label
        except:
            return item
    else:
        return item


def get_item_description(item):
    if re.match(r'[Q|P][0-9]{1,}', item):
        url = 'https://www.wikidata.org/wiki/Special:EntityData/' + item + '.json'
        r = requests.get(url, auth=('user', 'pass'))
        json_data = r.json()
        try:
            json_data['entities'][item]
        except KeyError:
            item = list(json_data['entities'].keys())[0]
        try:
            description = json_data['entities'][item]['descriptions']['en']['value']
        except KeyError:
            try:
                description = json_data['entities'][item]['descriptions']['de']['value']
            except KeyError:
                try:
                    description = json_data['entities'][item]['descriptions']['es']['value']
                except KeyError:
                    try:
                        description = json_data['entities'][item]['descriptions']['fr']['value']
                    except KeyError:
                        try:
                            description = json_data['entities'][item]['descriptions']['nl']['value']
                        except KeyError:
                            description = json_data['entities'][item]['descriptions']

        return description
    else:
        return item


sample['item_label'] = sample['item_id'].apply(lambda x: get_item_label(x))
sample['item_description'] = sample['item_id'].apply(lambda x: get_item_description(x))
sample['id'] = sample['reference_id'].apply(lambda x: hash(x))
sample['property_label'] = sample['stat_property'].apply(lambda x: get_item_label(x))
sample['value_label'] = sample['stat_value'].apply(lambda x: get_item_label(x))
sample['value_description'] = sample['stat_value'].apply(lambda x: get_item_description(x))

sample = sample[['id', 'item_id', 'stat_property', 'stat_value', 'item_label',
                 'item_description', 'property_label', 'value_label', 'value_description', 'ref_value']]