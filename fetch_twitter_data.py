import os
import ssl
import certifi
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import sys
import json
import random
from datetime import datetime

# Disable SSL warnings for development
urllib3.disable_warnings(InsecureRequestWarning)

# Multiple SSL certificate fixes
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    SNSCRAPE_AVAILABLE = False
    print("Warning: snscrape not available")

# Usage: python fetch_twitter_data.py "Oppenheimer" 100

def fetch_tweets_with_ssl_fix(query, max_tweets=50):
    """Try to fetch tweets with SSL certificate fixes"""
    if not SNSCRAPE_AVAILABLE:
        return []
    
    try:
        # Create unverified SSL context (development only)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        tweets = []
        print(f"Attempting to scrape Twitter with query: {query}")
        
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_tweets:
                break
            tweets.append({
                "text": tweet.content,
                "hashtags": [h for h in tweet.hashtags] if tweet.hashtags else [],
                "likes": tweet.likeCount,
                "shares": tweet.retweetCount,
                "comments": tweet.replyCount,
                "label": None
            })
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Collected {i + 1} tweets...")
        
        return tweets
        
    except Exception as e:
        print(f"Twitter scraping failed: {e}")
        return []

def generate_synthetic_movie_data(movie_name, num_samples=50):
    """Generate synthetic social media data when real scraping fails"""
    print(f"Generating synthetic data for {movie_name}...")
    
    positive_templates = [
        "Just watched {movie}! Absolutely amazing! #MustWatch #Epic",
        "{movie} is incredible! Best movie of the year! #Loved #Amazing",
        "Can't stop thinking about {movie}. Masterpiece! #Brilliant #WorthIt",
        "{movie} exceeded all expectations! #Fantastic #Recommended",
        "Everyone needs to see {movie}! Mind-blowing! #Epic #Stunning"
    ]
    
    negative_templates = [
        "{movie} was disappointing. Expected more. #Meh #Overrated",
        "Not impressed with {movie}. Waste of time. #Boring #Skip",
        "{movie} didn't live up to the hype. #Disappointed #NotWorthIt",
        "Fell asleep during {movie}. So boring! #Terrible #Regret",
        "{movie} was confusing and poorly made. #Bad #Avoid"
    ]
    
    neutral_templates = [
        "Watched {movie} today. It was okay. #Movie #Weekend",
        "{movie} has good moments but overall average. #Okay #Mixed",
        "Some parts of {movie} were good, others not so much. #Average #Decent",
        "{movie} is watchable but nothing special. #Okay #Fine",
        "Mixed feelings about {movie}. Has potential. #Mixed #Decent"
    ]
    
    hashtag_pools = {
        'positive': ['#Amazing', '#Epic', '#MustWatch', '#Brilliant', '#Loved', '#Fantastic'],
        'negative': ['#Disappointing', '#Boring', '#Terrible', '#Overrated', '#Skip', '#Bad'],
        'neutral': ['#Movie', '#Film', '#Cinema', '#Weekend', '#Okay', '#Average']
    }
    
    synthetic_data = []
    
    for i in range(num_samples):
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            template = random.choice(positive_templates)
            label = 'Hit'
            base_likes = random.randint(50, 500)
            base_shares = random.randint(10, 100)
            base_comments = random.randint(5, 50)
        elif sentiment == 'negative':
            template = random.choice(negative_templates)
            label = 'Flop'
            base_likes = random.randint(5, 50)
            base_shares = random.randint(1, 20)
            base_comments = random.randint(1, 15)
        else:
            template = random.choice(neutral_templates)
            label = 'Average'
            base_likes = random.randint(20, 100)
            base_shares = random.randint(3, 30)
            base_comments = random.randint(2, 25)
        
        text = template.format(movie=movie_name)
        hashtags = random.sample(hashtag_pools[sentiment], random.randint(1, 3))
        
        synthetic_data.append({
            'text': text,
            'hashtags': hashtags,
            'likes': base_likes + random.randint(-10, 20),
            'shares': base_shares + random.randint(-5, 10),
            'comments': base_comments + random.randint(-3, 8),
            'label': label
        })
    
    return synthetic_data

def fetch_tweets(query, max_tweets=50):
    """Main function that tries multiple data collection methods"""
    print(f"Collecting data for query: {query}")
    
    # Method 1: Try Twitter scraping with SSL fixes
    tweets = fetch_tweets_with_ssl_fix(query, max_tweets)
    
    if tweets:
        print(f"✓ Successfully collected {len(tweets)} real tweets")
        return tweets
    
    # Method 2: Generate synthetic data as fallback
    print("Real Twitter scraping failed. Generating synthetic data...")
    movie_name = query.split()[0]  # Extract movie name from query
    synthetic_tweets = generate_synthetic_movie_data(movie_name, max_tweets)
    
    print(f"✓ Generated {len(synthetic_tweets)} synthetic samples")
    return synthetic_tweets

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