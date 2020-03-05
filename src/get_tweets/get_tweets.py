import pickle

from TwitterAPI import TwitterAPI

from bld.project_paths import project_paths_join as ppj


# define query parameters
query = [
    {"q": "from:nytimes", "count": 100, "lang": "en", "tweet_mode": "extended"},
    {"q": "from:guardian", "count": 100, "lang": "en", "tweet_mode": "extended"},
]
endpoint = "search/tweets"  # https://api.twitter.com/1.1/search/tweets.json
# remember to specify your developing environment if you have one
# i.e. tweets/search/30day/my_env_name.json OR tweets/search/fullarchive/my_env_name.json


# RESTful request
def get_tweets(q):
    data = api.request(endpoint, q)
    return data


if __name__ == "__main__":

    # load Twittter credentials
    credentials = pickle.load(open(ppj("CREDENTIALS", "credentials.pickle"), "rb"))

    api = TwitterAPI(
        credentials["consumer_key"],
        credentials["consumer_secret"],
        credentials["access_token"],
        credentials["access_token_secret"],
    )
    # join found tweets in one list
    data = []
    for q in query:
        request = get_tweets(q)
        data.extend(request)

    with open(ppj("OUT_DATA", "tweets.pickle"), "wb") as out_file:
        pickle.dump(data, out_file)
