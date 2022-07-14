import os
import pandas as pd
from collections import Counter
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import tqdm
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

main_path = '/media/bigdata/nolme'
data_path = os.path.join(main_path, 'data')

# Load data
item_frame = pd.read_json(os.path.join(data_path, 'zotero_items.json'))

nlp = spacy.load('en_core_web_sm')
#doc = nlp(item_frame['full_text'].sum())
#
## all tokens that arent stop words or punctuations
##words = [token.text for token in doc
#words = [token.lemma_ for token in doc
#              if not token.is_stop and not token.is_punct]
## five most common tokens
#word_freq = Counter(words)
#common_words = word_freq.most_common(5)
#
#bag_of_words = item_frame['full_text'].apply(lambda x: x.split(' ')).sum()
#word_count = len(bag_of_words)
#word_count_dict = pd.DataFrame(Counter(bag_of_words))

# Parser for reviews
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

item_frame["processed_full_text"] = item_frame["full_text"].apply(spacy_tokenizer)

# Creating a vectorizer
vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, 
        stop_words='english', lowercase=True, 
        token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(item_frame["processed_full_text"])

NUM_TOPICS = 10

# Latent Dirichlet Allocation Model
lda = LatentDirichletAllocation(n_components=10, 
        max_iter=50, learning_method='online',verbose=True)
data_lda = lda.fit_transform(data_vectorized)

# Non-Negative Matrix Factorization Model
nmf = NMF(n_components=NUM_TOPICS)
data_nmf = nmf.fit_transform(data_vectorized) 

# Latent Semantic Indexing Model using Truncated SVD
lsi = TruncatedSVD(n_components=NUM_TOPICS)
data_lsi = lsi.fit_transform(data_vectorized)

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                for i in topic.argsort()[:-top_n - 1:-1]]) 

# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer)
