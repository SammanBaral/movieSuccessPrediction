"""
Alternative data collection methods for movie success prediction
when Twitter scraping fails due to SSL or API restrictions.
"""

import requests
import json
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for development (not recommended for production)
urllib3.disable_warnings(InsecureRequestWarning)

class MovieDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def generate_synthetic_data(self, movie_name, num_samples=100):
        """
        Generate synthetic social media data for testing when real data isn't available
        """
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
            'positive': ['#Amazing', '#Epic', '#MustWatch', '#Brilliant', '#Loved', '#Fantastic', '#Stunning', '#Masterpiece'],
            'negative': ['#Disappointing', '#Boring', '#Terrible', '#Overrated', '#Skip', '#Waste', '#Bad', '#Avoid'],
            'neutral': ['#Movie', '#Film', '#Cinema', '#Weekend', '#Okay', '#Average', '#Decent', '#Mixed']
        }
        
        synthetic_data = []
        
        for i in range(num_samples):
            # Randomly choose sentiment
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
    
    def collect_reddit_data(self, movie_name, subreddits=['movies', 'MovieReviews', 'boxoffice']):
        """
        Collect movie-related posts from Reddit (requires praw library)
        """
        try:
            import praw
            
            # You'll need to set up Reddit API credentials
            reddit = praw.Reddit(
                client_id="your_client_id",
                client_secret="your_client_secret",
                user_agent="movie_predictor_bot"
            )
            
            posts = []
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                for post in subreddit.search(movie_name, limit=50):
                    posts.append({
                        'text': f"{post.title} {post.selftext}",
                        'hashtags': [],  # Reddit doesn't use hashtags
                        'likes': post.score,
                        'shares': 0,  # Reddit doesn't have shares
                        'comments': post.num_comments,
                        'label': None
                    })
            
            return posts
        except ImportError:
            print("praw library not installed. Install with: pip install praw")
            return []
        except Exception as e:
            print(f"Reddit data collection failed: {e}")
            return []
    
    def collect_imdb_reviews(self, movie_name):
        """
        Scrape IMDB reviews (basic implementation)
        """
        try:
            # This is a simplified example - you'd need to implement proper IMDB scraping
            # or use their API if available
            print(f"IMDB scraping for {movie_name} would be implemented here")
            return []
        except Exception as e:
            print(f"IMDB scraping failed: {e}")
            return []
    
    def fix_ssl_twitter_scraping(self, movie_name, max_tweets=50):
        """
        Attempt to fix SSL issues with Twitter scraping
        """
        try:
            # Create unverified SSL context (not recommended for production)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Try alternative Twitter scraping with SSL fixes
            import snscrape.modules.twitter as sntwitter
            
            # Set environment variables for SSL
            import os
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['CURL_CA_BUNDLE'] = ''
            
            tweets = []
            query = f"{movie_name} lang:en since:2024-01-01"
            
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
            
            return tweets
            
        except Exception as e:
            print(f"SSL-fixed Twitter scraping failed: {e}")
            return []

def main():
    collector = MovieDataCollector()
    movie_name = "Oppenheimer"  # Example movie
    
    print(f"Collecting data for: {movie_name}")
    
    # Try different data collection methods
    methods = [
        ("Synthetic Data", lambda: collector.generate_synthetic_data(movie_name, 100)),
        ("Fixed SSL Twitter", lambda: collector.fix_ssl_twitter_scraping(movie_name, 50)),
        ("Reddit Data", lambda: collector.collect_reddit_data(movie_name))
    ]
    
    all_data = []
    
    for method_name, method_func in methods:
        print(f"\nTrying {method_name}...")
        try:
            data = method_func()
            if data:
                print(f"✓ {method_name}: Collected {len(data)} samples")
                all_data.extend(data)
            else:
                print(f"✗ {method_name}: No data collected")
        except Exception as e:
            print(f"✗ {method_name}: Failed - {e}")
    
    if all_data:
        # Save collected data
        df = pd.DataFrame(all_data)
        output_file = f"collected_data_{movie_name.replace(' ', '_')}.json"
        df.to_json(output_file, orient="records", indent=2)
        print(f"\n✓ Total collected: {len(all_data)} samples")
        print(f"✓ Saved to: {output_file}")
        
        # Show data distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"✓ Label distribution: {label_counts.to_dict()}")
    else:
        print("\n✗ No data collected from any method")

if __name__ == "__main__":
    main()
