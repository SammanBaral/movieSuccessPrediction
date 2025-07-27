import os
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
import snscrape.modules.twitter as sntwitter
import pandas as pd
import sys

# Usage: python fetch_twitter_data.py "Oppenheimer" 100

def fetch_tweets(query, max_tweets=50):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append({
            "text": tweet.content,
            "hashtags": [h for h in tweet.hashtags] if tweet.hashtags else [],
            "likes": tweet.likeCount,
            "shares": tweet.retweetCount,
            "comments": tweet.replyCount,
            "label": None  # To be labeled manually or semi-automatically
        })
    return tweets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_twitter_data.py <movie name> [max_tweets]")
        sys.exit(1)
    movie = sys.argv[1]
    max_tweets = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    # Example query: "Oppenheimer lang:en since:2024-01-01"
    query = f"{movie} lang:en since:2024-01-01"
    tweets = fetch_tweets(query, max_tweets=max_tweets)
    out_file = f"twitter_data_{movie.replace(' ', '_')}.json"
    pd.DataFrame(tweets).to_json(out_file, orient="records", indent=2)
    print(f"Saved {len(tweets)} tweets to {out_file}") 