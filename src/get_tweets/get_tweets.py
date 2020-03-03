import csv
import pickle

from TwitterAPI import TwitterAPI

from bld.project_paths import project_paths_join as ppj

# load Twittter credentials
credentials = pickle.load(open("src/credentials/credentials.pickle", "rb"))

api = TwitterAPI(
    credentials["consumer_key"],
    credentials["consumer_secret"],
    credentials["access_token"],
    credentials["access_token_secret"],
)

# define query parameters
query = {"q": "from:nytimes", "count": 50}


# RESTful request
# remember to specify your developing environment if you have one
# i.e. tweets/search/30day/my_env_name.json OR tweets/search/fullarchive/my_env_name.json
def get_tweets():
    r = api.request(  # https://api.twitter.com/1.1/search/tweets.json
        "search/tweets", query
    )
    data = []

    for item in r:
        data.append(item["text"])
        print(item["text"])

    return data


if __name__ == "__main__":
    data = get_tweets()

    with open(ppj("OUT_DATA", "tweets.csv"), "w", newline="") as out_file:
        wr = csv.writer(out_file, quoting=csv.QUOTE_ALL)
        wr.writerow(data)
