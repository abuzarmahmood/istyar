import os
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from pprint import pprint
import gensim.corpora as corpora
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import itertools as it
import numpy as np
import tqdm
from joblib import Parallel, delayed, cpu_count
import pyLDAvis.gensim
import pickle 
import pyLDAvis

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    mytokens = [ word.lemma_.lower().strip() for word in mytokens ]
            #if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens \
            if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           workers = 4)
    
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                    texts=processed_text, 
                                    dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

# Generate bi-grams
def gen_ngram(sentence, n=2):
    """
    Defaults to bigram
    """
    split_sent = sentence.split(' ')
    return [" ".join(split_sent[i:i+n]) for i in range(len(split_sent)-n+1)] 

base_dir = '/media/bigdata/projects/katz_pub_tree'
data_path = os.path.join(base_dir, 'data') 

paper_frame = pd.read_json(os.path.join(data_path,'katz_frame.json'))

# Simple preprocessing
abstracts = [x[0] for x in paper_frame['Abstract']]
#processed_abstracts = [gensim.utils.simple_preprocess(x) for x in abstracts]
processed_abstracts = [spacy_tokenizer(x) for x in abstracts]

titles = paper_frame['Title']
#processed_titles = [gensim.utils.simple_preprocess(x) for x in titles]
processed_titles = [spacy_tokenizer(x) for x in titles]

# Remove stopwords
#def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#    """https://spacy.io/api/annotation"""
#    texts_out = []
#    for sent in texts:
#        doc = nlp(" ".join(sent)) 
#        texts_out.append([token.lemma_ for token in doc \
#                        if token.pos_ in allowed_postags])
#    return texts_out

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, 
#        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


ngram_abstracts = [gen_ngram(x,1) for x in processed_abstracts]
ngram_titles = [gen_ngram(x,1) for x in processed_titles]

# Collect titles and abstracts
processed_text = [x+y for x,y in zip(ngram_titles, ngram_abstracts)]

# Create Dictionary
id2word = corpora.Dictionary(processed_text)
corpus = [id2word.doc2bow(text) for text in processed_text]

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, 
                                texts=processed_text, 
                                dictionary=id2word, 
                                coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

grid = {}
grid['Validation_Set'] = {}

# Topics range
topics_range = range(2, 11, 1)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3)) + ['symmetric','asymmetric']

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3)) + ['symmetric']

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run

iters = list(it.product(range(len(corpus_sets)), topics_range, alpha, beta))

#def parallelize(func, iterator):
#    return Parallel(n_jobs = cpu_count()-2)\
#            (delayed(func)(this_iter) for this_iter in tqdm.tqdm(iterator))
#
#def compute_coherence_par(params):
#    i,k,a,b = params
#    cv = compute_coherence_values(corpus=corpus_sets[i], 
#                            dictionary=id2word, 
#                                  k=k, a=a, b=b)
#    return cv
#
#out = parallelize(compute_coherence_par, iters) 

#pbar = tqdm.tqdm(total=len(iters))
#for i,k,a,b in iters:
#    # get the coherence score for the given parameters
#    cv = compute_coherence_values(corpus=corpus_sets[i], 
#                            dictionary=id2word, 
#                                  k=k, a=a, b=b)
#    # Save the model results
#    model_results['Validation_Set'].append(corpus_title[i])
#    model_results['Topics'].append(k)
#    model_results['Alpha'].append(a)
#    model_results['Beta'].append(b)
#    model_results['Coherence'].append(cv)
#    
#    pbar.update(1)

model_tuning_frame = pd.DataFrame(model_results)
model_tuning_frame.to_csv('lda_tuning_results.csv', index=False)
pbar.close()

if 'model_tuning_frame' not in globals().keys():
    model_tuning_frame = pd.read_csv('lda_tuning_results.csv')

best_params =  model_tuning_frame.sort_values(by='Coherence').tail(1)

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=int(best_params['Topics']), 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=best_params['Alpha'].values[0],
                                           eta=best_params['Beta'].values[0])

# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_prepared
