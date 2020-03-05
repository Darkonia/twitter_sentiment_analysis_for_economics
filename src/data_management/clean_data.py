import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from bld.project_paths import project_paths_join as ppj


def normalise_text(text_list, stopwords="true", lemmatise="true", stemming="false"):

    normalised_tweets = []

    for text in tweet_texts:
        normalised_row = clean_text(text)

        if stopwords == "true":
            normalised_row = remove_stopwords(normalised_row)
        if lemmatise == "true":
            normalised_row = lemmatize(normalised_row)
        if stemming == "true":
            normalised_row = stem(normalised_row)

        normalised_tweets.append(normalised_row)

    return normalised_tweets


def clean_text(text):
    """ Pre process and convert texts to a list of words
    method inspired by method from eliorc
    github repo: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb"""
    text = text.lower()

    # Clean the text
    text = re.sub(r"what[\',’]s", "what is ", text)
    text = re.sub(r"[\',’]s", " ", text)
    text = re.sub(r"[\',’]ve", " have ", text)
    text = re.sub(r"can[\',’]t", "cannot ", text)
    text = re.sub(r"n[\',’]t", " not ", text)
    text = re.sub(r"i[\',’]m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\-", " minus ", text)
    text = re.sub(r"\=", " equal ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)  #
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)  #
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"http.*", " ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # remove stopwords

    text = text.split()

    return text


def remove_stopwords(text, language="english"):
    stops = set(stopwords.words(language))
    without_stopwords = [x for x in text if x not in stops]
    return without_stopwords


stops = set(stopwords.words("english"))


# define methods stem and lemmatize for a list of words
def lemmatize(text):
    lmtzd_text = []
    lmtzr = WordNetLemmatizer()
    for word in text:
        lmtzd_word = lmtzr.lemmatize(word, get_wordnet_pos(word))
        lmtzd_text.append(lmtzd_word)
    return lmtzd_text


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def stem(text):
    stemd_text = []
    porter = PorterStemmer()
    for word in text:
        stem_word = porter.stem(word)
        stemd_text.append(stem_word)
    return stemd_text


if __name__ == "__main__":

    # load saved tweets from get_tweets.py
    data = pickle.load(open(ppj("TWEETS", "tweets.pickle"), "rb"))

    # extract only text from tweets
    tweet_texts = []
    for tweet in data:
        tweet_texts.append(tweet["full_text"])
    data = normalise_text(tweet_texts)
    with open(ppj("OUT_DATA", "normalised_tweets.pickle"), "wb") as out_file:
        pickle.dump(data, out_file)
