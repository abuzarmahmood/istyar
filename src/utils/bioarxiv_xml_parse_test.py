import os
import xmltodict
from pprint import pprint
from glob import glob
from time import sleep
import pubmed_parser as pp

def unit_parser(dat):
    wanted_keys = ['title','#text','sec','p']
    if isinstance(dat, dict):
        keys_present = [this_key for this_key in wanted_keys \
                if this_key in dat.keys()]
        wanted_secs = [dat[i] for i in keys_present]
        return wanted_secs
    elif isinstance(dat, list):
        parsed_list = [unit_parser(i) for i in dat]
        not_list = [x for x in parsed_list if not isinstance(x, list)]
        list_list = [x for x in parsed_list if isinstance(x, list)]
        flat_list_list = [x for y in list_list for x in y]
        return not_list + flat_list_list 
    elif isinstance(dat, str):
        return dat

def recurrent_parser(sections):
    parsed_dat = unit_parser(sections)
    dat_len = len(parsed_dat)
    #print(dat_len)
    while True:
        #sleep(0.1)
        parsed_dat =unit_parser(parsed_dat)
        new_len = len(parsed_dat)
        #print(new_len)
        if new_len == dat_len:
            break
        else:
            dat_len = new_len
    parsed_dat = [x for x in parsed_dat if isinstance(x, str)]
    return parsed_dat

def get_all_text(doc):
    meta_chunk = doc['article']['front']['article-meta']
    abstract = " ".join(recurrent_parser(meta_chunk['abstract']))
    body_sections = doc['article']['body']#['sec']
    body = recurrent_parser(body_sections)
    all_body  = " ".join(body)
    all_text = abstract + all_body
    return all_text

def get_article_metadata(doc):
    meta_chunk = doc['article']['front']['article-meta']
    title = unit_parser(meta_chunk['title-group']['article-title'])
    date_chunk = meta_chunk['pub-date']
    if isinstance(date_chunk, list) and len(date_chunk) > 1:
        date_chunk = date_chunk[0]
    date = date_chunk['year']
    #date = [date_chunk[x] for x in ['month','year'] if x in date_chunk.keys()]
    author_chunk = meta_chunk['contrib-group']['contrib']
    authors = [[x['name'][y] for y in ['given-names', 'surname']] \
            for x in author_chunk]
    authors = [" ".join(x) for x in authors]
    meta_dict = dict(
            title = title,
            date = date,
            authors = authors
            )
    return meta_dict


def return_doc(file_path):
    with open(file_path,'r') as fd:
        doc = xmltodict.parse(fd.read())
    return doc

############################################################

if __name__ == "__main__":
    #data_path = '/media/bigdata/projects/istyar/data/bioarxiv_s3'
    #file_list = glob(os.path.join(data_path,'*','*','content','*.xml'))

    data_path = '/media/bigdata/projects/istyar/data/plos'
    file_list = glob(os.path.join(data_path,'*.xml'))

    #dict_out = pp.parse_pubmed_xml(file_list[0])
    doc = return_doc(file_list[0])
    metadata = get_article_metadata(doc)
    all_text = get_all_text(doc)
