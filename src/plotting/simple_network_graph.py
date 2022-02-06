import os
import pandas as pd

base_dir = '/media/bigdata/projects/katz_pub_tree'
data_path = os.path.join(base_dir, 'data') 

paper_frame = pd.read_json(os.path.join(data_path,'katz_frame.json'))
