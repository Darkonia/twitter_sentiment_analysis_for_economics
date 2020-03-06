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


def replace_sentiment_words(word, sentiment_dict):
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


# document vectors with sentiment coefficients
def get_sentiment_coeff(data, sentiment_dict):
    """
    Replace each word with its associated sentiment coefficient from sentiment dict.
    Set coefficient to 0 if not found.
    """
    data_scores = []
    for tweet in data:
        tweet_scores = []
        for word in tweet:
            tweet_scores.append(replace_sentiment_words(word, sentiment_dict))

        data_scores.append(tweet_scores)
    return data_scores


# creates for a tweet a dict(word: tfidf_weight)
def create_tfidf_dict(doc_index, transformed, features):
    vector_coo = transformed[doc_index].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


# document vectors with tfidf weights
def get_tfidf_weights(data, tfidf_model):
    """
    Replace each word with its associated tfidf weight using tfidf_model.
    tfidf_model: TfidfVectorizer
    """
    data_weights = []
    counter = 0
    for tweet in data:
        tweet_weights = []
        tfidf_dict = create_tfidf_dict(counter, transformed, features)
        tfidf_dict
        for word in tweet:
            tweet_weights.append(tfidf_dict[word])
        data_weights.append(tweet_weights)
        counter += 1
    return data_weights


def compute_weighted_coeff(data_scores, data_weights):
    """
    Compute each docs weighted coefficient by adding the product of every word with its weight
    """
    data_weighted_coeff = []
    for tweet in range(len(data_scores)):
        data_weighted_coeff.append(np.dot(data_scores[tweet], data_weights[tweet]))
    return data_weighted_coeff


if __name__ == "__main__":

    # load normalised tweets
    data = pickle.load(open(ppj("OUT_DATA", "normalised_tweets.pickle"), "rb"))

    # Create CBOW model
    model = Word2Vec(data, min_count=2, size=300, window=4)

    word_vectors = model.wv

    # identify clusters using KMeans
    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(
        X=word_vectors.vectors
    )
    positive_cluster_center = model.cluster_centers_[0]
    negative_cluster_center = model.cluster_centers_[1]

    # dentify meaning of the cluster
    word_vectors.similar_by_vector(
        model.cluster_centers_[0], topn=10, restrict_vocab=None
    )

    word_vectors.similar_by_vector(
        model.cluster_centers_[1], topn=10, restrict_vocab=None
    )

    # summarise CBOW model
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

    # prepare tweets for tfidf
    tfidf_data = []

    for tweet in data:
        t_as_string = ""
        for word in tweet:
            t_as_string += word + " "
        tfidf_data.append(t_as_string[:-1])

    # tfidf
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(tfidf_data)

    transformed = tfidf.transform(tfidf_data)  # sparse matrix weight(doc, word)
    features = pd.Series(tfidf.get_feature_names())  # maps word to number

    # dict word to sentiment_coeff
    sentiment_dict = dict(zip(words["words"], words["sentiment_coeff"]))

    data_scores = get_sentiment_coeff(data, sentiment_dict)
    data_weights = get_tfidf_weights(data, tfidf)

    data_weighted_coeff = compute_weighted_coeff(data_scores, data_weights)

    with open(ppj("OUT_DATA", "data_weighted_coeff.pickle"), "wb") as out_file:
        pickle.dump(data_weighted_coeff, out_file)


def create_tfidf_dictionary(x, transformed_file, features):
    """
    create dictionary for each input sentence x, where each word has assigned its tfidf score

    inspired  by function from this wonderful article:
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles
    -using-nlp-bfd864f41b34

    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    """
    vector_coo = transformed[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


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
