import calendar
import pickle
import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from bld.project_paths import project_paths_join as ppj


def normalise_text(text, stopwords="true", lemmatise="true", stemming="false"):
    """Normalises the text using the selected methods in the parameters.
    returns a list of sentences, each as a list of words
    """
    normalised_row = clean_text(text)

    if stopwords == "true":
        normalised_row = remove_stopwords(normalised_row)
    if lemmatise == "true":
        normalised_row = lemmatize(normalised_row)
    if stemming == "true":
        normalised_row = stem(normalised_row)

    return normalised_row


def clean_text(text):
    """ Pre process and convert texts to a list of words.
    Formats and homogenises the text using regular expressions
    method nspired by method from eliorc
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
    text = re.sub(r":", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)  #
    text = re.sub(r" e g ", " eg ", text)  #
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"http.*", " ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"(?<!\S)\d(?![^\s.,?!])", " ", text)
    text = re.sub(r"(?<!\S)\d\d(?![^\s.,?!])", " ", text)
    text = re.sub(r"(?<!\S)\d\d\d(?![^\s.,?!])", " ", text)

    # remove stopwords

    text = text.split()

    return text


def remove_stopwords(text, language="english"):
    """removes words with no semantic value
    """
    stops = set(stopwords.words(language))
    without_stopwords = [x for x in text if x not in stops]
    return without_stopwords


stops = set(stopwords.words("english"))


# define methods stem and lemmatize for a list of words
def lemmatize(text):
    """reduce words using the semantic root
    """
    lmtzd_text = []
    lmtzr = WordNetLemmatizer()
    for word in text:
        lmtzd_word = lmtzr.lemmatize(word, get_wordnet_pos(word))
        lmtzd_text.append(lmtzd_word)
    return lmtzd_text


def get_wordnet_pos(word):
    """map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def stem(text):
    """reduce words to the smallest root. Roots do not necessarily have a meaning
    """
    stemd_text = []
    porter = PorterStemmer()
    for word in text:
        stem_word = porter.stem(word)
        stemd_text.append(stem_word)
    return stemd_text


# define periods to group tweets by date, in this case quarters
month_to_num = {v: "0" + str(k) for k, v in enumerate(calendar.month_abbr)}
month_to_num["Oct"] = 10
month_to_num["Nov"] = 11
month_to_num["Dec"] = 12


def reformat_date(tweet):
    date_list = tweet["created_at"].split()
    tweet["created_at"] = date_list[-1] + str(month_to_num[date_list[1]]) + date_list[2]
    return tweet["created_at"]


def identify_quarter(date):
    day = int(date[4:])
    qu = ""
    if day < 401:
        qu = "Q1"
    elif day < 701:
        qu = "Q2"
    elif day < 1001:
        qu = "Q3"
    elif day < 1301:
        qu = "Q4"
    return qu


def assign_quarter(tweet):
    """idetify tweet's date and assign respective quarter, example: 2020-03-08 -> 2020Q1
    """
    date = reformat_date(tweet)
    qu = identify_quarter(date)
    tweet["quarter"] = date[:4] + qu
    return tweet["quarter"]


if __name__ == "__main__":

    # load saved tweets from get_tweets.py
    data = pickle.load(open(ppj("OUT_DATA", "tweets.pickle"), "rb"))

    # creates dictionary by ordering tweets by quarter and save normalised text
    tweets_by_period = {}
    for tweet in data:
        assign_quarter(tweet)
        text = normalise_text(tweet["text"])
        try:
            tweets_by_period[tweet["quarter"]].append(text)
        except KeyError:
            tweets_by_period[tweet["quarter"]] = [text]

    with open(ppj("OUT_DATA", "normalised_tweets.pickle"), "wb") as out_file:
        pickle.dump(tweets_by_period, out_file)
