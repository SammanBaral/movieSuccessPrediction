"""
Enhanced Real Data Collection for Movie Success Prediction
Supports multiple data sources with robust error handling and rate limiting.
"""

import requests
import pandas as pd
import time
import json
import re
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import os
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MovieData:
    text: str
    hashtags: List[str]
    likes: int
    shares: int
    comments: int
    source: str
    timestamp: datetime
    movie_name: str
    label: Optional[str] = None

class EnhancedMovieDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def collect_reddit_data(self, movie_name: str, limit: int = 100) -> List[MovieData]:
        """
        Collect movie discussions from Reddit using PRAW
        """
        try:
            import praw
            
            # Reddit API credentials (you need to register at https://www.reddit.com/prefs/apps/)
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID', 'your_client_id'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'your_client_secret'),
                user_agent='movie_predictor_v1.0 by /u/yourusername'
            )
            
            data = []
            subreddits = ['movies', 'MovieReviews', 'boxoffice', 'film', 'moviecritic']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for posts about the movie
                    for submission in subreddit.search(movie_name, limit=limit//len(subreddits)):
                        # Extract text content
                        text = f"{submission.title}"
                        if submission.selftext:
                            text += f" {submission.selftext}"
                        
                        data.append(MovieData(
                            text=text,
                            hashtags=[],  # Reddit doesn't use hashtags
                            likes=submission.score,
                            shares=0,
                            comments=submission.num_comments,
                            source=f'reddit_{subreddit_name}',
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            movie_name=movie_name
                        ))
                        
                        # Add top comments
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments[:3]:  # Top 3 comments
                            if len(comment.body) > 20:  # Filter short comments
                                data.append(MovieData(
                                    text=comment.body,
                                    hashtags=[],
                                    likes=comment.score,
                                    shares=0,
                                    comments=len(comment.replies),
                                    source=f'reddit_{subreddit_name}_comment',
                                    timestamp=datetime.fromtimestamp(comment.created_utc),
                                    movie_name=movie_name
                                ))
                        
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logger.warning(f"Error collecting from r/{subreddit_name}: {e}")
                    continue
                    
            logger.info(f"Collected {len(data)} posts from Reddit")
            return data
            
        except ImportError:
            logger.error("PRAW not installed. Install with: pip install praw")
            return []
        except Exception as e:
            logger.error(f"Reddit data collection failed: {e}")
            return []
    
    def collect_imdb_reviews(self, movie_name: str, imdb_id: str = None) -> List[MovieData]:
        """
        Collect reviews from IMDb using web scraping
        """
        try:
            from bs4 import BeautifulSoup
            
            if not imdb_id:
                # Search for the movie first
                search_url = f"https://www.imdb.com/find?q={movie_name.replace(' ', '+')}"
                response = self.session.get(search_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the first movie result
                movie_link = soup.find('a', href=re.compile(r'/title/tt\d+/'))
                if movie_link:
                    imdb_id = re.search(r'tt\d+', movie_link['href']).group()
                else:
                    logger.warning(f"Could not find IMDb ID for {movie_name}")
                    return []
            
            # Get reviews
            reviews_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
            response = self.session.get(reviews_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = []
            review_containers = soup.find_all('div', class_='review-container')
            
            for container in review_containers[:50]:  # Limit to 50 reviews
                try:
                    title_elem = container.find('a', class_='title')
                    content_elem = container.find('div', class_='text')
                    rating_elem = container.find('span', class_='rating-other-user-rating')
                    
                    if title_elem and content_elem:
                        title = title_elem.get_text(strip=True)
                        content = content_elem.get_text(strip=True)
                        
                        # Extract rating if available
                        rating = 0
                        if rating_elem:
                            rating_text = rating_elem.get_text(strip=True)
                            rating_match = re.search(r'(\d+)', rating_text)
                            if rating_match:
                                rating = int(rating_match.group(1))
                        
                        data.append(MovieData(
                            text=f"{title} {content}",
                            hashtags=[],
                            likes=rating * 10,  # Convert rating to likes-like metric
                            shares=0,
                            comments=0,
                            source='imdb_review',
                            timestamp=datetime.now(),
                            movie_name=movie_name
                        ))
                        
                except Exception as e:
                    logger.warning(f"Error parsing review: {e}")
                    continue
                    
                time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Collected {len(data)} reviews from IMDb")
            return data
            
        except ImportError:
            logger.error("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"IMDb scraping failed: {e}")
            return []
    
    def collect_youtube_comments(self, movie_name: str) -> List[MovieData]:
        """
        Collect comments from YouTube movie trailers/reviews
        Note: Requires YouTube Data API key
        """
        try:
            api_key = os.getenv('YOUTUBE_API_KEY')
            if not api_key:
                logger.warning("YouTube API key not found. Set YOUTUBE_API_KEY environment variable")
                return []
            
            # Search for movie trailers
            search_url = "https://www.googleapis.com/youtube/v3/search"
            search_params = {
                'part': 'snippet',
                'q': f"{movie_name} trailer",
                'type': 'video',
                'maxResults': 5,
                'key': api_key
            }
            
            response = self.session.get(search_url, params=search_params)
            search_data = response.json()
            
            data = []
            
            for item in search_data.get('items', []):
                video_id = item['id']['videoId']
                
                # Get comments for this video
                comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
                comments_params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'maxResults': 20,
                    'key': api_key
                }
                
                comments_response = self.session.get(comments_url, params=comments_params)
                comments_data = comments_response.json()
                
                for comment_item in comments_data.get('items', []):
                    comment = comment_item['snippet']['topLevelComment']['snippet']
                    
                    data.append(MovieData(
                        text=comment['textDisplay'],
                        hashtags=[],
                        likes=comment['likeCount'],
                        shares=0,
                        comments=comment_item['snippet']['totalReplyCount'],
                        source='youtube_comment',
                        timestamp=datetime.fromisoformat(comment['publishedAt'].replace('Z', '+00:00')),
                        movie_name=movie_name
                    ))
                
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Collected {len(data)} comments from YouTube")
            return data
            
        except Exception as e:
            logger.error(f"YouTube data collection failed: {e}")
            return []
    
    def collect_news_articles(self, movie_name: str) -> List[MovieData]:
        """
        Collect movie-related news articles using NewsAPI
        """
        try:
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key:
                logger.warning("News API key not found. Set NEWS_API_KEY environment variable")
                return []
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': movie_name,
                'sortBy': 'popularity',
                'language': 'en',
                'pageSize': 50,
                'apiKey': api_key
            }
            
            response = self.session.get(url, params=params)
            news_data = response.json()
            
            data = []
            
            for article in news_data.get('articles', []):
                if article['title'] and article['description']:
                    text = f"{article['title']} {article['description']}"
                    
                    data.append(MovieData(
                        text=text,
                        hashtags=[],
                        likes=0,  # News articles don't have likes
                        shares=0,
                        comments=0,
                        source='news_article',
                        timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                        movie_name=movie_name
                    ))
            
            logger.info(f"Collected {len(data)} news articles")
            return data
            
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            return []
    
    def collect_all_data(self, movie_name: str, imdb_id: str = None) -> List[MovieData]:
        """
        Collect data from all available sources
        """
        logger.info(f"Starting comprehensive data collection for: {movie_name}")
        
        all_data = []
        
        # Collect from each source
        sources = [
            ("Reddit", lambda: self.collect_reddit_data(movie_name)),
            ("IMDb", lambda: self.collect_imdb_reviews(movie_name, imdb_id)),
            ("YouTube", lambda: self.collect_youtube_comments(movie_name)),
            ("News", lambda: self.collect_news_articles(movie_name))
        ]
        
        for source_name, collect_func in sources:
            try:
                logger.info(f"Collecting from {source_name}...")
                source_data = collect_func()
                all_data.extend(source_data)
                logger.info(f"✓ {source_name}: {len(source_data)} items collected")
            except Exception as e:
                logger.error(f"✗ {source_name}: Collection failed - {e}")
        
        logger.info(f"Total collected: {len(all_data)} items from all sources")
        return all_data
    
    def save_data(self, data: List[MovieData], filename: str):
        """
        Save collected data to JSON file
        """
        # Convert to dictionary format
        data_dict = []
        for item in data:
            data_dict.append({
                'text': item.text,
                'hashtags': item.hashtags,
                'likes': item.likes,
                'shares': item.shares,
                'comments': item.comments,
                'source': item.source,
                'timestamp': item.timestamp.isoformat(),
                'movie_name': item.movie_name,
                'label': item.label
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")

def main():
    """
    Example usage of the enhanced data collector
    """
    collector = EnhancedMovieDataCollector()
    
    # Example: Collect data for Oppenheimer
    movie_name = "Oppenheimer"
    imdb_id = "tt15398776"  # Optional: IMDb ID for more accurate results
    
    # Collect all data
    data = collector.collect_all_data(movie_name, imdb_id)
    
    if data:
        # Save to file
        filename = f"real_data_{movie_name.lower().replace(' ', '_')}.json"
        collector.save_data(data, filename)
        
        # Print summary
        print(f"\n📊 Data Collection Summary for {movie_name}")
        print("=" * 50)
        print(f"Total items collected: {len(data)}")
        
        # Source breakdown
        sources = {}
        for item in data:
            sources[item.source] = sources.get(item.source, 0) + 1
        
        for source, count in sources.items():
            print(f"  {source}: {count} items")
        
        # Sample data
        print(f"\n📝 Sample Data:")
        for i, item in enumerate(data[:3]):
            print(f"{i+1}. [{item.source}] {item.text[:100]}...")
    else:
        print("No data collected. Please check your API keys and network connection.")

if __name__ == "__main__":
    main()
