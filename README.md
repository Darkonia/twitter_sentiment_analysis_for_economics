# twitter_sentiment_analysis_for_economics
Sentiment analysis template optimised for the use of twitter data in economic contexts.

## Motivation
Twitter contains a tons of interesting information for economic research. Sadly, there aren't many labeled data sets that suit economic research points of interest.
Therefore, an unsupervised machine learning method is needed instead of starting labelling data.

## Features
Using Twitter API, this package allows you to connect your Twitter Developer account, define a query and instantly get the tweets.
Basic steps for Text Normalisation are already included and are easy to adapt to your research project's requirements.
By Clustering the word2vec Vectors, identify the sentiment of words and assign a weigthed Coefficient to each tweet.
Get a first report what defines each cluster/sentiment and basic models using the created coefficient. 
Analyse the regression graphs and tables from the automatic analysis.

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Requirements
Conda environment in ennvironment.yml
Twitter Developer account and respective credentials if you want to make your own requests.

## Tests
Working on MacOS Sierra Version 10.13.6
Testing data cleaning process (text normalisation). *if you adapt this process to your own requirements, remeber to dissable or update the tests

## How to use?
First, save your Twitter Developer credentials in secrets.py (and encrypt them).
Then define your queries in get_tweets.py.
You can change a few parameters how the sentiment coefficient for each tweet is computed in clean_data.py and sentiment_model.py.
Then, load you own data to regress on the sentiment coefficient in analysis.py.
Check the automatically created reports and see if you found interesting relations!

Or to recreate my project just run it in waf.

## Credits
Thanks to the internet community for making awesome content available;)

#### Anything else that seems useful

## License
GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007
Sentiment analysis package optimised for the use of twitter data in economic contexts
Copyright (C) 2020  Joaquin Felber
