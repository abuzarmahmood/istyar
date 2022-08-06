# Imports

import os
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import numpy as np
from tqdm import tqdm
import pylab as plt
from sklearn.decomposition import PCA as pca
import umap
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances as distmat
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import plotly.express as px
from plotly.subplots import make_subplots

# Load data
base_dir = '/media/bigdata/projects/istyar'
data_path = os.path.join(base_dir, 'data') 
file_list = glob(os.path.join(data_path,"*.json"))
file_list = [x for x in file_list if '/author_frame.json' not in x]

# Exctract data needed for processin
wanted_cols = ['Date', 'processed_titles', 'PMID', 'processed_abstracts', 'Authors']
paper_frame = pd.read_json(file_list[0])[wanted_cols]
for file_path in tqdm(file_list[1:]):
    paper_frame = pd.concat([paper_frame, pd.read_json(file_path)[wanted_cols]])
processed_text = paper_frame['processed_titles'] + ' ' + paper_frame['processed_abstracts']

# Extract word vectors
max_features = 2000
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=max_features, stop_words="english").fit(processed_text)
tfidf = tfidf_vectorizer.transform(processed_text)

# Extract topics
total_topics = 20
nmf = NMF(n_components=total_topics, random_state=1, alpha=0.1, l1_ratio=0.5, verbose=1).fit(tfidf)
nmf_embed = nmf.transform(tfidf)

# Perform PCA to reduce dimensions
# Take components that give x% variance explained
pca_object = pca().fit(nmf_embed)
var_thresh = 0.95
n_components = np.where(np.cumsum(pca_object.explained_variance_ratio_) >= var_thresh)[0][0]
plt.plot(np.cumsum(pca_object.explained_variance_ratio_),'-x')
plt.axvline(n_components, color = 'red')

fin_pca_obj = pca(n_components = n_components).fit(nmf_embed)
pca_embeddings = fin_pca_obj.transform(nmf_embed)

# Perform same processing pipeline on data subset for clustering
main_frame = pd.read_json('/media/bigdata/projects/istyar/data/katz_frame.json')
main_frame['processed_text'] = main_frame['processed_titles'] + ' ' + main_frame['processed_abstracts']
tfidf_mean = tfidf_vectorizer.transform(main_frame['processed_text'])
nmf_embed_main = nmf.transform(tfidf_mean)
pca_embeddings_main = fin_pca_obj.transform(nmf_embed_main)

# Find close connections
# From the distribution of ALL DISTANCES, find those which are whithin some threshold
unique_dists = np.tril(dists)
unique_dists = unique_dists[np.where(unique_dists)]
thresh = 5 # th percentile of closest distances
cutoff = np.percentile(unique_dists, thresh)
mask_array = dists <= cutoff
diag_inds = np.diag_indices(mask_array.shape[0])
mask_array[diag_inds] = 0
fin_similarity = dists.copy()
fin_similarity = fin_similarity - np.min(fin_similarity)
fin_similarity = fin_similarity / np.max(fin_similarity)
fin_similarity = 1 - fin_similarity
fin_similarity[~mask_array] = 0
#plt.imshow(fin_similarity, aspect='auto')

n_clusters = 5
fin_model = GMM(n_components=n_clusters, random_state=0).fit(pca_embeddings_main)
predictions = fin_model.predict(pca_embeddings_main)

main_frame['group'] = predictions
#main_frame['group_fin'] = main_frame['group'] -1 + np.random.random(len(main_frame['group']))*0.5

# Calculate edges for graphs
# y_anchors = np.arange(fin_model.n_components+1) - 0.5
# y_edges = [y_anchors[i:i+2] for i in range(len(y_anchors)-1)]
# x_edges = [main_frame['Date'].min()-1, main_frame['Date'].max()+1]

# Walk through points in the same group
# Push them apart on the y-axis until they are some threshold apart
fin_group_list = []
grouped_frame = list(main_frame.groupby('group'))
num,this_frame = grouped_frame[0]
for num, this_frame in grouped_frame:
    subgrouped_frame = list(this_frame.groupby('Date'))
    for subnum, this_subgrouped_frame in subgrouped_frame:
        #subnum, this_subgrouped_frame = subgrouped_frame[0]
        sub_len = len(this_subgrouped_frame)
        if sub_len > 1:
            this_subgrouped_frame['group_fin'] = \
                this_subgrouped_frame['group'] + np.linspace(-0.3,0.3, sub_len)
        else:
            this_subgrouped_frame['group_fin'] = \
                this_subgrouped_frame['group']
        fin_group_list.append(this_subgrouped_frame)
main_frame = pd.concat(fin_group_list)

# Calculate edges between nodes
undirected_adjacency = np.tril(mask_array)
x = main_frame['Date'].values
y = main_frame['group_fin'].values

# Find most prominent topic per group
num_groups = np.sort(np.unique(predictions))
group_inds = [np.where(predictions==x)[0] for x in num_groups]
embed_main_grouped = [nmf_embed_main[x] for x in group_inds]
embed_main_max = [np.argmax(np.mean(x,axis=0)) for x in embed_main_grouped]

word_weight_group_list = [x.dot(nmf.components_) for x in embed_main_grouped]
mean_word_weight_group = [np.mean(x,axis=0) for x in word_weight_group_list]
word_weight_group_mean_rank = [np.argsort(x) for x in mean_word_weight_group]
top = 5
top_group_words_inds = [x[-top:] for x in word_weight_group_mean_rank]
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
feature_array = np.array(tfidf_feature_names)
top_group_words = [feature_array[np.array(x)] for x in top_group_words_inds]
fin_group_word_weight = [[top_group_words[group], mean_word_weight_group[group][top_group_words_inds[group]]]\
                         for group in num_groups]

def get_link_inds(ind):
    return list(np.where(undirected_adjacency[ind])[0]) + list(np.where(undirected_adjacency.T[ind])[0])
def get_edges(x,y,ind0,ind1):
    return [x[ind0],y[ind0],x[ind1],y[ind1]]

inds = np.where(undirected_adjacency)

edge_list = []
for ind0,ind1 in zip(inds[0],inds[1]):
    edge_list.append(get_edges(x,y,ind0,ind1))


# Plotting
row_num = len(num_groups)

def specs_gen(row_num):
    row1=[{}, {"rowspan" : row_num, "colspan" : 4},None,None,None]
    row_template = [{}, None,None,None,None]
    remaining_rows = [row_template for i in range(row_num-1)]
    fin_list = [row1] + remaining_rows
    return fin_list

np.random.seed(1)

#unique_article_types = paper_frame['Fin_Article_Type'].unique()
#color_map = dict(zip(unique_article_types,range(len(unique_article_types))))

f = make_subplots(
    rows=row_num, cols=5,
    specs=specs_gen(row_num))
    #subplot_titles=("First Subplot","Second Subplot", "Third Subplot"))

# f.add_trace(go.Scatter(x=[1, 2], y=[1, 2]),
#                  row=1, col=2)
# f.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]),
#                  row=2, col=1)

x=main_frame["Date"]
y=main_frame["group_fin"]
#color = [color_map[x] for x in main_frame['Fin_Article_Type']]
size = main_frame['scatter_size']
#custom_text = main_frame['Title_Pretty']
#text = paper_frame['Summary']
text = main_frame["Title"]

#f = go.FigureWidget([go.Scatter(x=x, y=y, 
f.add_trace(go.Scatter(x=x, y=y, 
                                mode='markers',
                               hovertemplate =
                                '<b>%{text}</b>',
                                text = text),
           row=1, col=2)

# Add word bar charts
for this_num in num_groups:
    words, height = fin_group_word_weight[this_num][0], fin_group_word_weight[this_num][1]
    f.add_trace(go.Bar(x = words, y = height),#, orientation='h'),
               row = row_num-this_num, col = 1)
    #f.update_layout(xaxis_tickangle=-45)

f.update_layout(
    autosize=False,
    width=1000,
    height=1000)

pio.write_html(f, file='grouped_articles_with_bars.html', auto_open=False)