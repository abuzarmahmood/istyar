import os
import xmltodict
from glob import glob

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

