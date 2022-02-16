from Bio import Entrez
import json
import pandas as pd
import datetime
import os
from tqdm import tqdm
import time
import numpy as np
import uuid
import sys
from glob import glob

base_dir = '/media/bigdata/projects/istyar'
query_dir = os.path.join(base_dir, 'src','query')
sys.path.append(query_dir)
os.chdir(query_dir)
from query_helper import *

data_path = os.path.join(base_dir, 'data') 
author_frame_path = os.path.join(data_path, 'author_frame.json')

#  ___                        
# / _ \ _   _  ___ _ __ _   _ 
#| | | | | | |/ _ \ '__| | | |
#| |_| | |_| |  __/ |  | |_| |
# \__\_\\__,_|\___|_|   \__, |
#                       |___/ 

author_name = 'Pulakat, Lakshmi'
affiliation = None
and_str = " AND "

punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
# Removing punctuations in string
# Using loop + punctuation string
for ele in author_name:
    if ele in punc:
        author_name = author_name.replace(ele, "")

if author_name:
    query = f'({author_name}[Author])'
    if affiliation:
        query = query + and_str + f'({affiliation}[Affiliation])'

author_id = str(uuid.uuid4()).split('-')[0]

processed_name = author_name.lower()
processed_name = processed_name.replace(' ', '_')

author_dict = {'name' : processed_name, 'id' : author_id}

if not os.path.exists(author_frame_path):
    temp_frame = pd.DataFrame(author_dict, index = [0])
    temp_frame.to_json(author_frame_path)
else:
    # Load frame, check presence of author, write out accordingly
    temp_frame = pd.read_json(author_frame_path)
    #if any(temp_frame['name'].str.contains(processed_name)):
    #    raise Exception('Author name already present in database')
    #else:
    #    temp_frame = temp_frame.append(author_dict, ignore_index=True)
    #    temp_frame.to_json(author_frame_path)



#    _         _   _                
#   / \  _   _| |_| |__   ___  _ __ 
#  / _ \| | | | __| '_ \ / _ \| '__|
# / ___ \ |_| | |_| | | | (_) | |   
#/_/   \_\__,_|\__|_| |_|\___/|_|   
                                   

results = search(query)
id_list = results['IdList']
papers = fetch_details(id_list)

parsed_paper_list = [return_article_attrs(x) for x in papers['PubmedArticle']]
parsed_paper_list = [x for num,x in enumerate(parsed_paper_list) if x is not None]
paper_frame = pd.DataFrame(parsed_paper_list)
paper_frame.dropna(inplace=True)

# Print query and ask if this what they want 
pretty_authors = [x[0].split()[-1] + ' et al.' for x in paper_frame['Authors']]
paper_frame['pretty_authors'] = pretty_authors

paper_frame[['Date','Title','pretty_authors']]

print(' === DONE GRABBING AUTHOR ARTICLES ===')
paper_frame.to_parquet(
        os.path.join(data_path,f'{processed_name}_author_frame.pq'))

# Check if current papers match saved ones
data_files = glob(os.path.join(data_path, "*.pq")) 
all_ids_frame = [pd.read_parquet(x, columns = ['PMID']) for x in tqdm(data_files)]
all_ids_set = set()
for this_frame in all_ids_frame:
    #this_frame.iloc[:,0].to_list()
    for this_val in this_frame.iloc[:,0].to_list():
        all_ids_set.add(this_val)

for this_id in paper_frame['PMID']:
    if float(this_id) in all_ids_set:
        print(True)


## Save to parquet
#basenames = [x.split('.')[0] for x in data_files]
#for this_dat, this_name in zip(all_ids, basenames):
#    this_dat.to_parquet(this_name+".pq")

#  ____ _ _        _   _                 
# / ___(_) |_ __ _| |_(_) ___  _ __  ___ 
#| |   | | __/ _` | __| |/ _ \| '_ \/ __|
#| |___| | || (_| | |_| | (_) | | | \__ \
# \____|_|\__\__,_|\__|_|\___/|_| |_|___/
#                                        

citations_frame = pd.concat([citation_parser(x) for x in papers['PubmedArticle']])
citations_frame.reset_index(drop=True, inplace=True)
citations_frame.drop_duplicates(inplace=True)

unique_citation_ids = [str(x) for x in citations_frame['ArticleIdList'].unique()]
results = fetch_details(unique_citation_ids)['PubmedArticle']
parsed_citations_list = [return_article_attrs(x) for x in results]
parsed_citations_list = [x for num,x in enumerate(parsed_citations_list) if x is not None]
parsed_citations_frame = pd.DataFrame(parsed_citations_list)
parsed_citations_frame.dropna(inplace=True)

fin_citations_frame = citations_frame.merge(parsed_citations_frame,
                        left_on = 'ArticleIdList', right_on = 'PMID')
fin_citations_frame.drop(columns = 'ArticleIdList',inplace=True)

print(' === DONE GRABBING CITATIONS ===')
fin_citations_frame.to_parquet(
        os.path.join(data_path,f'{processed_name}_citations_frame.pq'))

# ____  _           _ _            
#/ ___|(_)_ __ ___ (_) | __ _ _ __ 
#\___ \| | '_ ` _ \| | |/ _` | '__|
# ___) | | | | | | | | | (_| | |   
#|____/|_|_| |_| |_|_|_|\__,_|_|   
#                                  

def get_similar_frame(pmid):
    try:
        #time.sleep(np.random.random()*2)
        record = Entrez.read(Entrez.elink(dbfrom='pubmed', id = pmid))
        similar_ids = [x['Id'] for x in record[0]["LinkSetDb"][0]["Link"]]
        similar_papers = fetch_details(similar_ids)['PubmedArticle']
        similar_papers_list = [return_article_attrs(x) for x in similar_papers]
        similar_papers_list = [x for num,x in enumerate(similar_papers_list) \
                if x is not None]
        similar_papers_frame = pd.DataFrame(similar_papers_list)
        return similar_papers_frame
    except:
        print('Something went wrong...couldnt load similar article')
        return None

og_and_cite_unique_ids = \
        list(set([*fin_citations_frame['PMID'], *paper_frame['PMID']]))
og_and_cite_unique_ids = [str(x) for x in og_and_cite_unique_ids]

fin_similar_list = [get_similar_frame(x) for x in tqdm(og_and_cite_unique_ids)] 
fin_similar_list = [x for x in fin_similar_frame if x is not None]

for unique_id, frame in zip(og_and_cite_unique_ids, fin_similar_list):
    frame['similar_to'] = unique_id
#fin_similar_list = parallelize(get_similar_frame, og_and_cite_unique_ids) 
fin_similar_frame = pd.concat(fin_similar_list).reset_index(drop=True)

print(' === DONE GRABBING SIMILAR ARTICLES ===')
fin_similar_frame.to_parquet(
        os.path.join(data_path,f'{processed_name}_similar_frame.pq'))
