from Bio import Entrez
import json
import pandas as pd
import datetime
import os
from tqdm import tqdm
import time
import numpy as np
#import xmltodict

base_dir = '/media/bigdata/projects/katz_pub_tree'
data_path = os.path.join(base_dir, 'data') 
API_KEY = '244721b223855e6ed42690a047212dbbe408' 


def search(query):
    Entrez.email = 'abuzarmahmood@gmail.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='100',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'abuzarmahmood@gmail.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids, 
                           api_key =API_KEY)
    results = Entrez.read(handle)
    return results

def author_list_parser(author_list):
    #wanted_attrs = ['ForeName','Initials','LastName']
    wanted_attrs = ['ForeName','LastName']
    #return [" ".join((x[wanted_attrs[0]], x[wanted_attrs[1]],x[wanted_attrs[2]]))\
    return [" ".join((x[wanted_attrs[0]], x[wanted_attrs[1]]))\
                    for x in author_list]

def date_parser(date):
    #if len(date) > 0:
    try:
        return date['Year']
        #return "-".join(list(date.values()))
    except:
        return None


def return_article_attrs(paper):
    #wanted_attrs = {'MedlineCitation' :
    #                {'Article' : ['ArticleDate','ArticleTitle',
    #                    {'Abstract' : 'AbstractText'},
    #                        'AuthorList']}} 

    # Hardcode indexing for now
    try:
        #date_raw = paper['MedlineCitation']['Article']['ArticleDate']
        date_raw = paper['MedlineCitation']['Article']\
                                ['Journal']['JournalIssue']['PubDate']
        date_parsed = date_parser(date_raw) 
        title = paper['MedlineCitation']['Article']['ArticleTitle']
        pmid = paper['MedlineCitation']['PMID']
        article_type = paper['MedlineCitation']['Article']["PublicationTypeList"]
        abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText']
        author_list = paper['MedlineCitation']['Article']['AuthorList']
        parsed_authors = author_list_parser(author_list)
        attrs_keys = ['Date','Title','PMID',"Article_Type",
                        'Abstract','Authors']

        article_series = pd.Series(dict(zip(attrs_keys, 
            [date_parsed,title,pmid,article_type,abstract,parsed_authors])))
        return article_series
    except:
        return None

def return_citation_attrs(parsed_citation):


def citation_parser(paper):
    try:
        paper_pmid = str(return_article_attrs(paper)[2])
        citations = paper['PubmedData']['ReferenceList'][0]['Reference']
        citations_frame = pd.DataFrame(citations)
        citations_frame['ArticleIdList'] = [x[0] for x in citations_frame['ArticleIdList']]
        citations_frame['cited_by'] = paper_pmid
        return citations_frame
    except:
        pass

def pprint(x):
    print(json.dumps(x, indent=4))

#    _         _   _                
#   / \  _   _| |_| |__   ___  _ __ 
#  / _ \| | | | __| '_ \ / _ \| '__|
# / ___ \ |_| | |_| | | | (_) | |   
#/_/   \_\__,_|\__|_| |_|\___/|_|   
                                   

results = search('(Katz DB[Author]) AND (Brandeis[Affiliation])')
id_list = results['IdList']
papers = fetch_details(id_list)
#for i, paper in enumerate(papers['PubmedArticle']):
#     print("{}) {}".format(i+1, 
#         paper['MedlineCitation']['Article']['ArticleTitle']))

# Pretty print the first paper in full to observe its structure
#print(json.dumps(papers['PubmedArticle'][0], indent=4))

#x = papers['PubmedArticle'][0]

parsed_paper_list = [return_article_attrs(x) for x in papers['PubmedArticle']]
parsed_paper_list = [x for num,x in enumerate(parsed_paper_list) if x is not None]
paper_frame = pd.DataFrame(parsed_paper_list)
paper_frame.dropna(inplace=True)


paper_frame.to_json(os.path.join(data_path,'katz_frame.json'))

#  ____ _ _        _   _                 
# / ___(_) |_ __ _| |_(_) ___  _ __  ___ 
#| |   | | __/ _` | __| |/ _ \| '_ \/ __|
#| |___| | || (_| | |_| | (_) | | | \__ \
# \____|_|\__\__,_|\__|_|\___/|_| |_|___/
#                                        

citations_frame = pd.concat([citation_parser(x) for x in papers['PubmedArticle']])
citations_frame.reset_index(drop=True, inplace=True)
citations_frame.drop_duplicates(inplace=True)

#pmid = str(citations_frame.iloc[0,1])
#handle = Entrez.efetch(db="pubmed", id=pmid, retmode = 'xml')
#test = Entrez.read(handle)
#test = test['PubmedArticle'][0]
#handle.close()

unique_citation_ids = [str(x) for x in citations_frame['ArticleIdList'].unique()]
#handle = Entrez.efetch(db="pubmed", id=unique_citation_ids, retmode = 'xml')
#results = Entrez.read(handle)['PubmedArticle']
results = fetch_details(unique_citation_ids)['PubmedArticle']
parsed_citations_list = [return_article_attrs(x) for x in results]
parsed_citations_list = [x for num,x in enumerate(parsed_citations_list) if x is not None]
parsed_citations_frame = pd.DataFrame(parsed_citations_list)
parsed_citations_frame.dropna(inplace=True)

fin_citations_frame = citations_frame.merge(parsed_citations_frame,
                        left_on = 'ArticleIdList', right_on = 'PMID')
fin_citations_frame.drop(columns = 'ArticleIdList',inplace=True)

fin_citations_frame.to_json(os.path.join(data_path,'katz_citations_frame.json'))

# ____  _           _ _            
#/ ___|(_)_ __ ___ (_) | __ _ _ __ 
#\___ \| | '_ ` _ \| | |/ _` | '__|
# ___) | | | | | | | | | (_| | |   
#|____/|_|_| |_| |_|_|_|\__,_|_|   
#                                  

#from joblib import Parallel, delayed, cpu_count
def parallelize(func, iterator):
    return Parallel(n_jobs = 3)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def get_similar_frame(pmid):
    #time.sleep(np.random.random()*2)
    record = Entrez.read(Entrez.elink(dbfrom='pubmed', id = pmid, api_key = API_KEY))
    similar_ids = [x['Id'] for x in record[0]["LinkSetDb"][0]["Link"]]
    similar_papers = fetch_details(similar_ids)['PubmedArticle']
    similar_papers_list = [return_article_attrs(x) for x in similar_papers]
    similar_papers_list = [x for num,x in enumerate(similar_papers_list) if x is not None]
    similar_papers_frame = pd.DataFrame(similar_papers_list)
    return similar_papers_frame

og_and_cite_unique_ids = list(set([*fin_citations_frame['PMID'], *paper_frame['PMID']]))
og_and_cite_unique_ids = [str(x) for x in og_and_cite_unique_ids]
fin_similar_list = [get_similar_frame(x) for x in tqdm(og_and_cite_unique_ids)] 
for unique_id, frame in zip(og_and_cite_unique_ids, fin_similar_list):
    frame['similar_to'] = unique_id
#fin_similar_list = parallelize(get_similar_frame, og_and_cite_unique_ids) 
fin_similar_frame = pd.concat(fin_similar_list).reset_index(drop=True)
fin_similar_frame.to_json(os.path.join(data_path,'katz_similar_frame.json'))
