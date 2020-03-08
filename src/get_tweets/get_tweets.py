import pickle
import time

from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterRequestError

from bld.project_paths import project_paths_join as ppj


# define query parameters
time_periods = [
    201504011400,
    201507011400,
    201610011400,
    201601010000,
    201604011400,
    201607011400,
    201610011400,
    201701011400,
]
start = 201401011400
# (fromDate, toDate) format is YYYYMMDDhhmm. Doesn't apply for standard search
t_per_query = 25
query = [
    {
        "query": "from:federalreserve lang:en",
        "maxResults": t_per_query,
        "fromDate": start,
        "toDate": 0,
    },
    {
        "query": "from:elonmusk lang:en",
        "maxResults": t_per_query,
        "fromDate": start,
        "toDate": 0,
    },
    {
        "query": "from:HouseDemocrats lang:en",
        "maxResults": t_per_query,
        "fromDate": start,
        "toDate": 0,
    },
    {
        "query": "from:HouseDemocrats lang:en",
        "maxResults": t_per_query,
        "fromDate": start,
        "toDate": 0,
    },
    {
        "query": "from:HouseGOP lang:en",
        "maxResults": t_per_query,
        "fromDate": start,
        "toDate": 0,
    },
]
endpoint = (
    "tweets/search/fullarchive/:test"  # https://api.twitter.com/1.1/search/tweets.json
)
# remember to specify your developing environment if you have one
# i.e. tweets/search/30day/my_env_name.json OR tweets/search/fullarchive/my_env_name.json


# RESTful request
def get_tweets(query, endpoint, credentials):

    api = TwitterAPI(
        credentials["consumer_key"],
        credentials["consumer_secret"],
        credentials["access_token"],
        credentials["access_token_secret"],
    )

    # join found tweets in one list
    data = []
    for t in range(len(time_periods)):

        for q in query:
            time.sleep(1)  # avoid too many queries at once
            q["toDate"] = time_periods[t]
            request = api.request(endpoint, q)
            data.extend(request)
    return data


if __name__ == "__main__":
    data = []
    try:
        # load Twittter credentials
        credentials = pickle.load(open(ppj("CREDENTIALS", "credentials.pickle"), "rb"))
        data = get_tweets(query, endpoint, credentials)
        # save in original_data to skip this steps in following builds
        with open(ppj("IN_DATA", "tweets.pickle"), "wb") as out_file:
            pickle.dump(data, out_file)
    except TwitterRequestError:
        print("working with stored data!")
        # if request fails, load stored data
        data = pickle.load(open(ppj("IN_DATA", "tweets.pickle"), "rb"))

    # save data
    with open(ppj("OUT_DATA", "tweets.pickle"), "wb") as out_file:
        pickle.dump(data, out_file)
