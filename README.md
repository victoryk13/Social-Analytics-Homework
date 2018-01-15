

```python
# Dependencies
import tweepy
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = 'mTxrsDzw4ScwUl3W5ArZycZ8t'
consumer_secret = '1wjhOqXGx9kAMtJpA9kFPJeQ2O0ylIpXjT2RKvmOybzTmk31Ia'
access_token = '3019161335-6Q25TAuZqik8LdpoJuhMMsBE6tjOF6Nwcy9rnXI'
access_token_secret = 'jkbiWXzIj2Po8Rwqkyvu68BDhEAkvVzasK2luLGdUDXcI'

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Account
target_user_cnn = "@cnnbrk"

# Variables for holding sentiments
compound_list_cnn = []
positive_list_cnn = []
negative_list_cnn = []
neutral_list_cnn = []
source_account_cnn = []
tweet_text_cnn = []
tweet_date_cnn = []
tweet_count_cnn = list(range(1, 101))

# Loop through 5 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user_cnn, page=x)

    # Loop through all tweets
    for tweet in public_tweets:
        source_account_cnn.append(tweet["user"]["name"])
        tweet_text_cnn.append(tweet["text"])
        tweet_date_cnn.append(tweet["created_at"])
        
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate list
        compound_list_cnn.append(compound)
        positive_list_cnn.append(pos)
        negative_list_cnn.append(neg)
        neutral_list_cnn.append(neu)
        
cnn_df = pd.DataFrame(
    {'Source Account': source_account_cnn,
     'Tweet Text': tweet_text_cnn,
     'Tweet Date': tweet_date_cnn,
     'Compound Score': compound_list_cnn,
     'Positive Score': positive_list_cnn,
     'Neutral Score': neutral_list_cnn,
     'Negative Score': negative_list_cnn
    })

cnn_df = cnn_df[["Source Account", "Tweet Text", "Tweet Date", "Compound Score", "Positive Score", "Neutral Score", "Negative Score"]]
```


```python
# Target User Account
target_user_bbc = "@BBCBreaking"

# Variables for holding sentiments
compound_list_bbc = []
positive_list_bbc = []
negative_list_bbc = []
neutral_list_bbc = []
source_account_bbc = []
tweet_text_bbc = []
tweet_date_bbc = []
tweet_count_bbc = list(range(1, 101))

# Loop through 5 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user_bbc, page=x)

    # Loop through all tweets
    for tweet in public_tweets:
        source_account_bbc.append(tweet["user"]["name"])
        tweet_text_bbc.append(tweet["text"])
        tweet_date_bbc.append(tweet["created_at"])

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate list
        compound_list_bbc.append(compound)
        positive_list_bbc.append(pos)
        negative_list_bbc.append(neg)
        neutral_list_bbc.append(neu)
        
bbc_df = pd.DataFrame(
    {'Source Account': source_account_bbc,
     'Tweet Text': tweet_text_bbc,
     'Tweet Date': tweet_date_bbc,
     'Compound Score': compound_list_bbc,
     'Positive Score': positive_list_bbc,
     'Neutral Score': neutral_list_bbc,
     'Negative Score': negative_list_bbc
    })

bbc_df = bbc_df[["Source Account", "Tweet Text", "Tweet Date", "Compound Score", "Positive Score", "Neutral Score", "Negative Score"]]
```


```python
# Target User Account
target_user_cbs = "@CBS"

# Variables for holding sentiments
compound_list_cbs = []
positive_list_cbs = []
negative_list_cbs = []
neutral_list_cbs = []
source_account_cbs = []
tweet_text_cbs = []
tweet_date_cbs = []
tweet_count_cbs = list(range(1, 101))

# Loop through 5 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user_cbs, page=x)

    # Loop through all tweets
    for tweet in public_tweets:
        source_account_cbs.append(tweet["user"]["name"])
        tweet_text_cbs.append(tweet["text"])
        tweet_date_cbs.append(tweet["created_at"])

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate list
        compound_list_cbs.append(compound)
        positive_list_cbs.append(pos)
        negative_list_cbs.append(neg)
        neutral_list_cbs.append(neu)
        
cbs_df = pd.DataFrame(
    {'Source Account': source_account_cbs,
     'Tweet Text': tweet_text_cbs,
     'Tweet Date': tweet_date_cbs,
     'Compound Score': compound_list_cbs,
     'Positive Score': positive_list_cbs,
     'Neutral Score': neutral_list_cbs,
     'Negative Score': negative_list_cbs
    })

cbs_df = cbs_df[["Source Account", "Tweet Text", "Tweet Date", "Compound Score", "Positive Score", "Neutral Score", "Negative Score"]]
```


```python
# Target User Account
target_user_fox = "@FoxNews"

# Variables for holding sentiments
compound_list_fox = []
positive_list_fox = []
negative_list_fox = []
neutral_list_fox = []
source_account_fox = []
tweet_text_fox = []
tweet_date_fox = []
tweet_count_fox = list(range(1, 101))

# Loop through 5 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user_fox, page=x)

    # Loop through all tweets
    for tweet in public_tweets:
        source_account_fox.append(tweet["user"]["name"])
        tweet_text_fox.append(tweet["text"])
        tweet_date_fox.append(tweet["created_at"])

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate list
        compound_list_fox.append(compound)
        positive_list_fox.append(pos)
        negative_list_fox.append(neg)
        neutral_list_fox.append(neu)
        
fox_df = pd.DataFrame(
    {'Source Account': source_account_fox,
     'Tweet Text': tweet_text_fox,
     'Tweet Date': tweet_date_fox,
     'Compound Score': compound_list_fox,
     'Positive Score': positive_list_fox,
     'Neutral Score': neutral_list_fox,
     'Negative Score': negative_list_fox
    })

fox_df = fox_df[["Source Account", "Tweet Text", "Tweet Date", "Compound Score", "Positive Score", "Neutral Score", "Negative Score"]]
```


```python
# Target User Account
target_user_nyt = "@nytimes"

# Variables for holding sentiments
compound_list_nyt = []
positive_list_nyt = []
negative_list_nyt = []
neutral_list_nyt = []
source_account_nyt = []
tweet_text_nyt = []
tweet_date_nyt = []
tweet_count_nyt = list(range(1, 101))

# Loop through 5 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user_nyt, page=x)

    # Loop through all tweets
    for tweet in public_tweets:
        source_account_nyt.append(tweet["user"]["name"])
        tweet_text_nyt.append(tweet["text"])
        tweet_date_nyt.append(tweet["created_at"])

        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]

        # Add each value to the appropriate list
        compound_list_nyt.append(compound)
        positive_list_nyt.append(pos)
        negative_list_nyt.append(neg)
        neutral_list_nyt.append(neu)
        
nyt_df = pd.DataFrame(
    {'Source Account': source_account_nyt,
     'Tweet Text': tweet_text_nyt,
     'Tweet Date': tweet_date_nyt,
     'Compound Score': compound_list_nyt,
     'Positive Score': positive_list_nyt,
     'Neutral Score': neutral_list_nyt,
     'Negative Score': negative_list_nyt
    })

nyt_df = nyt_df[["Source Account", "Tweet Text", "Tweet Date", "Compound Score", "Positive Score", "Neutral Score", "Negative Score"]]
```


```python
seaborn.set()
plt.scatter(tweet_count_bbc, compound_list_bbc, label="BBC", c="turquoise", edgecolors="black")
plt.scatter(tweet_count_cbs, compound_list_cbs, label="CBS", c="green", edgecolors="black")
plt.scatter(tweet_count_cnn, compound_list_cnn, label="CNN", c="red", edgecolors="black")
plt.scatter(tweet_count_fox, compound_list_fox, label="Fox", c="blue", edgecolors="black")
plt.scatter(tweet_count_nyt, compound_list_nyt, label="New York Times", c="yellow", edgecolors="black")
plt.title("Sentiment Analysis of Media Tweets (01/14/2018)")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.yticks(np.arange(-1, 1.5, 0.5))
plt.legend(title="Media Sources", bbox_to_anchor=(1.04,1), loc="upper left")
plt.gca().invert_xaxis()
plt.savefig("SentimentAnalysisScatterPlot.png")
plt.show()
```


![png](output_6_0.png)



```python
news_outlets = ["BBC", "CBS", "CNN", "Fox", "NYT"]
x_axis = np.arange(0, len(news_outlets))
tick_locations = []
for x in x_axis:
    tick_locations.append(x + 1)
fig, ax = plt.subplots()  

seaborn.set()
plt.title("Overall Media Sentiment based on Twitter (01/14/2018)")
plt.ylabel("Tweet Polarity")
plt.xlim(0.5,5.5)
plt.bar(1, np.mean(compound_list_bbc), width = 1, facecolor="turquoise")
plt.bar(2, np.mean(compound_list_cbs), width = 1, facecolor="green")
plt.bar(3, np.mean(compound_list_cnn), width = 1, facecolor="red")
plt.bar(4, np.mean(compound_list_fox), width = 1, facecolor="blue")
plt.bar(5, np.mean(compound_list_nyt), width = 1, facecolor="yellow")
plt.xticks(tick_locations, news_outlets)
ax.text(0.75,-0.31, str(round(np.mean(compound_list_bbc),2)))
ax.text(1.8,0.35, str(round(np.mean(compound_list_cbs),2)))
ax.text(2.75,-0.18, str(round(np.mean(compound_list_cnn),2)))
ax.text(3.75,-0.07, str(round(np.mean(compound_list_fox),2)))
ax.text(4.75,-0.09, str(round(np.mean(compound_list_nyt),2)))
plt.grid(b=None)
plt.savefig("SentimentAnalysisBarChart.png")
plt.show()
```


![png](output_7_0.png)



```python
combined_df = bbc_df.append(cbs_df)
combined_df = combined_df.append(cnn_df)
combined_df = combined_df.append(fox_df)
combined_df = combined_df.append(nyt_df)
combined_df.to_csv("SentimentAnalysisData.csv")
```
