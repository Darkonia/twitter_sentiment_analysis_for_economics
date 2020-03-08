import pickle

from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRequestError

from bld.project_paths import project_paths_join as ppj

# define query parameters
time = (
    20200215000,
    202002200000,
)  # (201801010000, 201912312359)
# (fromDate, toDate) format is YYYYMMDDhhmm. Doesn't apply for standard search

query = [
    {
        "query": "from:federalreserve lang:en",
        "maxResults": 100,
        "fromDate": time[0],
        "toDate": time[1],
    },
    {
        "query": "from:elonmusk lang:en",
        "maxResults": 100,
        "fromDate": time[0],
        "toDate": time[1],
    },
    {
        "query": "from:HouseDemocrats lang:en",
        "maxResults": 100,
        "fromDate": time[0],
        "toDate": time[1],
    },
    {
        "query": "from:HouseDemocrats lang:en",
        "maxResults": 100,
        "fromDate": time[0],
        "toDate": time[1],
    },
    {
        "query": "from:HouseGOP lang:en",
        "maxResults": 100,
        "fromDate": time[0],
        "toDate": time[1],
    },
]
endpoint = (
    "tweets/search/30day/:test1"  # https://api.twitter.com/1.1/search/tweets.json
)
# remember to specify your developing environment if you have one
# i.e. tweets/search/30day/my_env_name.json OR tweets/search/fullarchive/my_env_name.json


# RESTful request
def get_tweets(q):
    data = api.request(endpoint, q)
    return data


if __name__ == "__main__":
    try:
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
        # save in original_data to skip this steps in following builds
        with open(ppj("IN_DATA", "tweets.pickle"), "wb") as out_file:
            pickle.dump(data, out_file)
    except TwitterRequestError:
        # if request fails, load stored data
        data = pickle.load(open(ppj("IN_DATA", "tweets.pickle"), "rb"))
        print()

    with open(ppj("OUT_DATA", "tweets.pickle"), "wb") as out_file:
        pickle.dump(data, out_file)
