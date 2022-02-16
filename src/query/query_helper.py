from Bio import Entrez
import json
import pandas as pd
import datetime
import os
from tqdm import tqdm
import time
import numpy as np
import uuid

# _   _      _                   _____                 _   _                 
#| | | | ___| |_ __   ___ _ __  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
#| |_| |/ _ \ | '_ \ / _ \ '__| | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#|  _  |  __/ | |_) |  __/ |    |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#|_| |_|\___|_| .__/ \___|_|    |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#             |_|                                                            

entrez_email = 'abuzarmahmood@gmail.com'

def search(query):
    Entrez.email = entrez_email
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='100',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = entrez_email
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids) 
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

