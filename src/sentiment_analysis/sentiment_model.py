# Python program to generate word vectors using Word2Vec
# importing all necessary modules
import pickle
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from bld.project_paths import project_paths_join as ppj

warnings.filterwarnings(action="ignore")  # check later

# Reads normalised tweets
data = pickle.load(open(ppj("OUT_DATA", "normalised_tweets.pickle"), "rb"))

# Create CBOW model
model1 = Word2Vec(data, min_count=2, size=300, window=4)

word_vectors = model1.wv


# identify clusters using KMeans
model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(
    X=word_vectors.vectors
)
positive_cluster_center = model.cluster_centers_[0]
negative_cluster_center = model.cluster_centers_[1]

word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)

word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)


words = pd.DataFrame(word_vectors.vocab.keys())
words.columns = ["words"]
words["vectors"] = words.words.apply(lambda x: word_vectors.wv[f"{x}"])
words["cluster"] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])
words["cluster_value"] = [1 if i == 0 else -1 for i in words.cluster]
words["closeness_score"] = words.apply(
    lambda x: 1 / (model.transform([x.vectors]).min()), axis=1
)
words["sentiment_coeff"] = words.closeness_score * words.cluster_value

# Print results

print(
    "Cosine similarity between 'alice' " + "and 'wonderland' - CBOW : ",
    model1.similarity("female", "male"),
)


# prepare tweets for tfidf
tfidf_data = []

for tweet in data:
    t_as_string = ""
    for word in tweet:
        t_as_string += word + " "
    tfidf_data.append(t_as_string)

# tfidf
tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(tfidf_data)
features = pd.Series(tfidf.get_feature_names())  # maps word to number
transformed = tfidf.transform(
    tfidf_data
)  # gives words and its tfidf weight for each tweet


# create doc sentiment_index using tfidf weights and sentiment coefficients
a = transformed.tocoo()
a.col = features[a.col]
print(a)
a.data


# def create_tfidf_dictionary(x, transformed_file, features):
#     """
#     create dictionary for each input sentence x, where each word has assigned its tfidf score
#
#     inspired  by function from this wonderful article:
#     https://medium.com/analytics-vidhya/automated-keyword-extraction-
# 	from-articles-using-nlp-bfd864f41b34
#
#     x - row of dataframe, containing sentences, and their indexes,
#     transformed_file - all sentences transformed with TfidfVectorizer
#     features - names of all words in corpus used in TfidfVectorizer
#     """
#     vector_coo = transformed_file[x.name].tocoo()
#     vector_coo.col = features.iloc[vector_coo.col].values
#     dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
#     return dict_from_coo
#
#
# def replace_tfidf_words(x, transformed_file, features):
#     """
#     replacing each word with it's calculated tfidf dictionary with scores of each word
#     x - row of dataframe, containing sentences, and their indexes,
#     transformed_file - all sentences transformed with TfidfVectorizer
#     features - names of all words in corpus used in TfidfVectorizer
#     """
#     dictionary = create_tfidf_dictionary(x, transformed_file, features)
