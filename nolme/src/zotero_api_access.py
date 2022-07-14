from pyzotero import zotero
import os
import pandas as pd
import numpy as np

main_path = '/media/bigdata/nolme'
config_path = os.path.join(main_path, 'config') 
data_path = os.path.join(main_path, 'data')

# Personal library
api_key = open(os.path.join(config_path, 'zotero_api_key')).read().strip()
library_id = open(os.path.join(config_path, 'zotero_library_id')).read().strip()
library_type = 'user'

zot = zotero.Zotero(library_id, library_type, api_key)
items = zot.all_top()

# Pull out Title and Abstract for each item
item_dat = [{"title" : x['data']['title'], 
                "abstract" : x['data']['abstractNote']} for x in items \
                        if 'abstractNote' in x['data'].keys()]

item_frame = pd.DataFrame(item_dat)
item_frame['full_text'] =  item_frame['title'] + " " + item_frame['abstract']

# Write out data
item_frame.to_json(os.path.join(data_path,'zotero_items.json'))

## we've retrieved the latest five top-level items in our library
## we can print each item's item type and ID
#for item in items:
#    print('Item: %s | Key: %s' % (item['data']['itemType'], item['data']['key']))

# Public library
#library_id = open(os.path.join(config_path, 'zotero_library_id')).read().strip()
#library_type = 'user'
library_id = '14500'
library_type = 'group'

zot = zotero.Zotero(library_id, library_type)
items = zot.all_top()

# Pull out Title and Abstract for each item
item_dat = [{"title" : x['data']['title'].strip(), 
                "abstract" : x['data']['abstractNote'].strip()} for x in items \
                        if 'abstractNote' in x['data'].keys()]

item_frame = pd.DataFrame(item_dat)
item_frame = item_frame[item_frame['abstract'] != '']
item_frame['full_text'] =  item_frame['title'] + " " + item_frame['abstract']

# Write out data
item_frame.to_json(os.path.join(data_path,f'library_{library_id}_items.json'))
