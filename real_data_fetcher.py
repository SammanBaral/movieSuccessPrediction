"""
Real Movie Data Fetcher
Fetches actual movie data from TMDB API and other sources
"""

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import os
from typing import List, Dict, Any

class RealMovieDataFetcher:
    def __init__(self, tmdb_api_key: str = None):
        # TMDB API setup (free tier available)
        self.tmdb_api_key = tmdb_api_key or "YOUR_TMDB_API_KEY"  # Replace with actual key
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        
        # Setup headers
        self.session.headers.update({
            'User-Agent': 'MovieSuccessPrediction/1.0',
            'Accept': 'application/json'
        })
    
    def get_popular_movies(self, pages: int = 10) -> List[Dict]:
        """Fetch popular movies from TMDB"""
        movies = []
        
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/popular"
                params = {
                    'api_key': self.tmdb_api_key,
                    'page': page,
                    'language': 'en-US'
                }
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    movies.extend(data.get('results', []))
                    print(f"✅ Fetched page {page}: {len(data.get('results', []))} movies")
                    time.sleep(0.25)  # Rate limiting
                else:
                    print(f"❌ Error fetching page {page}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error fetching page {page}: {e}")
                
        return movies
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Get detailed information about a specific movie"""
        try:
            url = f"{self.tmdb_base_url}/movie/{movie_id}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'credits,reviews,keywords,videos'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error fetching movie {movie_id}: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Error fetching movie {movie_id}: {e}")
            return {}
    
    def get_box_office_data(self, movies: List[Dict]) -> List[Dict]:
        """Enrich movie data with box office information"""
        enriched_movies = []
        
        for movie in movies:
            try:
                movie_details = self.get_movie_details(movie['id'])
                if movie_details:
                    # Extract relevant information
                    enriched_movie = {
                        'id': movie['id'],
                        'title': movie.get('title', ''),
                        'original_title': movie.get('original_title', ''),
                        'overview': movie.get('overview', ''),
                        'release_date': movie.get('release_date', ''),
                        'popularity': movie.get('popularity', 0),
                        'vote_average': movie.get('vote_average', 0),
                        'vote_count': movie.get('vote_count', 0),
                        'genre_ids': movie.get('genre_ids', []),
                        'budget': movie_details.get('budget', 0),
                        'revenue': movie_details.get('revenue', 0),
                        'runtime': movie_details.get('runtime', 0),
                        'adult': movie.get('adult', False),
                        'backdrop_path': movie.get('backdrop_path', ''),
                        'poster_path': movie.get('poster_path', ''),
                    }
                    
                    # Add success classification
                    enriched_movie['success_category'] = self.classify_success(enriched_movie)
                    
                    enriched_movies.append(enriched_movie)
                    print(f"✅ Processed: {enriched_movie['title']}")
                    
                time.sleep(0.25)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error processing movie: {e}")
                
        return enriched_movies
    
    def classify_success(self, movie: Dict) -> str:
        """Classify movie success based on revenue, ratings, and popularity"""
        revenue = movie.get('revenue', 0)
        budget = movie.get('budget', 0)
        rating = movie.get('vote_average', 0)
        popularity = movie.get('popularity', 0)
        
        # Calculate profit ratio
        profit_ratio = revenue / budget if budget > 0 else 0
        
        # Classification logic
        if profit_ratio >= 3 and rating >= 7.5 and popularity >= 50:
            return 'Hit'
        elif profit_ratio >= 1.5 and rating >= 6.0 and popularity >= 20:
            return 'Average'
        else:
            return 'Flop'
    
    def generate_synthetic_social_data(self, movies: List[Dict]) -> List[Dict]:
        """Generate realistic social media data based on actual movie metrics"""
        import random
        import numpy as np
        
        social_data = []
        
        for movie in movies:
            # Base social metrics on actual movie success
            success = movie['success_category']
            popularity = movie.get('popularity', 0)
            rating = movie.get('vote_average', 0)
            
            # Generate social media data for different time periods
            for days_before in range(1, 366, 5):  # Every 5 days for a year
                
                # Base engagement calculation
                base_engagement = 0.1
                if success == 'Hit':
                    base_engagement = 0.6 + (popularity / 200) + (rating / 20)
                elif success == 'Average':
                    base_engagement = 0.3 + (popularity / 400) + (rating / 40)
                else:
                    base_engagement = 0.1 + (popularity / 800) + (rating / 80)
                
                # Time-based multiplier (closer to release = more buzz)
                time_multiplier = 1.0
                if days_before <= 30:
                    time_multiplier = 2.0
                elif days_before <= 60:
                    time_multiplier = 1.5
                elif days_before <= 120:
                    time_multiplier = 1.2
                
                # Add randomness
                noise = random.uniform(0.8, 1.2)
                engagement_rate = min(1.0, base_engagement * time_multiplier * noise)
                
                # Generate social media metrics
                base_posts = int(popularity * time_multiplier * random.uniform(0.5, 2.0))
                
                social_record = {
                    'movie_id': movie['id'],
                    'movie_name': movie['title'],
                    'genre': self.get_genre_name(movie.get('genre_ids', [None])[0]),
                    'days_before_release': days_before,
                    'engagement_rate': round(engagement_rate, 4),
                    'post_count': max(1, base_posts),
                    'likes': int(base_posts * engagement_rate * random.uniform(10, 50)),
                    'shares': int(base_posts * engagement_rate * random.uniform(2, 15)),
                    'comments': int(base_posts * engagement_rate * random.uniform(1, 8)),
                    'buzz_type': random.choice(['Social Media', 'News', 'Trailer', 'Reviews', 'Celebrity']),
                    'label': success,
                    'actual_budget': movie.get('budget', 0),
                    'actual_revenue': movie.get('revenue', 0),
                    'actual_rating': movie.get('vote_average', 0),
                    'actual_popularity': movie.get('popularity', 0)
                }
                
                social_data.append(social_record)
        
        return social_data
    
    def get_genre_name(self, genre_id: int) -> str:
        """Convert genre ID to name"""
        genre_map = {
            28: 'Action', 35: 'Comedy', 18: 'Drama', 27: 'Horror',
            10749: 'Romance', 878: 'Sci-Fi', 53: 'Thriller', 16: 'Animation',
            80: 'Crime', 99: 'Documentary', 10751: 'Family', 14: 'Fantasy',
            36: 'History', 10402: 'Music', 9648: 'Mystery', 10770: 'TV Movie',
            37: 'Western', 12: 'Adventure', 10752: 'War'
        }
        return genre_map.get(genre_id, 'Unknown')
    
    def fetch_real_data(self, use_cache: bool = True) -> List[Dict]:
        """Main method to fetch and process real movie data"""
        cache_file = 'real_movie_data_cache.json'
        
        # Check cache first
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                print(f"📁 Loaded {len(cached_data)} records from cache")
                return cached_data
            except:
                print("❌ Cache file corrupted, fetching fresh data")
        
        print("🌐 Fetching real movie data from TMDB API...")
        
        # Check if API key is provided
        if self.tmdb_api_key == "YOUR_TMDB_API_KEY":
            print("⚠️  No TMDB API key provided, using sample real data")
            return self.create_sample_real_data()
        
        # Fetch popular movies
        movies = self.get_popular_movies(pages=5)  # ~100 movies
        print(f"📊 Fetched {len(movies)} popular movies")
        
        # Enrich with detailed data
        enriched_movies = self.get_box_office_data(movies[:20])  # Limit to prevent API overuse
        print(f"💰 Enriched {len(enriched_movies)} movies with box office data")
        
        # Generate social media data
        social_data = self.generate_synthetic_social_data(enriched_movies)
        print(f"📱 Generated {len(social_data)} social media records")
        
        # Cache the results
        try:
            with open(cache_file, 'w') as f:
                json.dump(social_data, f, indent=2)
            print(f"💾 Cached data to {cache_file}")
        except Exception as e:
            print(f"❌ Failed to cache data: {e}")
        
        return social_data
    
    def create_sample_real_data(self) -> List[Dict]:
        """Create sample data based on real movie characteristics"""
        import random
        
        # Real recent movies with actual data
        real_movies = [
            {'title': 'Avatar: The Way of Water', 'budget': 460000000, 'revenue': 2320250281, 'rating': 7.6, 'genre': 'Sci-Fi', 'success': 'Hit'},
            {'title': 'Top Gun: Maverick', 'budget': 170000000, 'revenue': 1488732821, 'rating': 8.3, 'genre': 'Action', 'success': 'Hit'},
            {'title': 'Black Panther: Wakanda Forever', 'budget': 250000000, 'revenue': 859208423, 'rating': 6.7, 'genre': 'Action', 'success': 'Average'},
            {'title': 'Jurassic World Dominion', 'budget': 265000000, 'revenue': 1006027065, 'rating': 5.6, 'genre': 'Action', 'success': 'Average'},
            {'title': 'Doctor Strange in the Multiverse of Madness', 'budget': 200000000, 'revenue': 956775804, 'rating': 6.9, 'genre': 'Action', 'success': 'Hit'},
            {'title': 'Minions: The Rise of Gru', 'budget': 80000000, 'revenue': 939422112, 'rating': 6.5, 'genre': 'Animation', 'success': 'Hit'},
            {'title': 'Thor: Love and Thunder', 'budget': 250000000, 'revenue': 760928081, 'rating': 6.2, 'genre': 'Action', 'success': 'Average'},
            {'title': 'The Batman', 'budget': 185000000, 'revenue': 771269432, 'rating': 7.8, 'genre': 'Action', 'success': 'Hit'},
            {'title': 'Sonic the Hedgehog 2', 'budget': 110000000, 'revenue': 405421518, 'rating': 6.5, 'genre': 'Animation', 'success': 'Average'},
            {'title': 'Fantastic Beasts: The Secrets of Dumbledore', 'budget': 200000000, 'revenue': 405201772, 'rating': 6.2, 'genre': 'Fantasy', 'success': 'Flop'},
            {'title': 'Morbius', 'budget': 75000000, 'revenue': 174003668, 'rating': 5.1, 'genre': 'Action', 'success': 'Flop'},
            {'title': 'The Northman', 'budget': 70000000, 'revenue': 69478160, 'rating': 7.0, 'genre': 'Drama', 'success': 'Flop'}
        ]
        
        social_data = []
        
        for movie in real_movies:
            # Calculate realistic social metrics based on success
            for days_before in range(1, 366, 3):
                success = movie['success']
                
                # Base engagement based on actual performance
                if success == 'Hit':
                    base_engagement = random.uniform(0.5, 0.9)
                elif success == 'Average':
                    base_engagement = random.uniform(0.2, 0.6)
                else:
                    base_engagement = random.uniform(0.05, 0.3)
                
                # Time multiplier
                if days_before <= 30:
                    time_mult = random.uniform(1.5, 2.5)
                elif days_before <= 90:
                    time_mult = random.uniform(1.0, 1.8)
                else:
                    time_mult = random.uniform(0.3, 1.2)
                
                engagement = min(1.0, base_engagement * time_mult)
                
                # Calculate posts based on budget and success
                budget_factor = movie['budget'] / 100000000  # Normalize to 100M
                base_posts = int(budget_factor * 100 * time_mult * random.uniform(0.5, 2.0))
                
                social_record = {
                    'movie_id': hash(movie['title']) % 10000,
                    'movie_name': movie['title'],
                    'genre': movie['genre'],
                    'days_before_release': days_before,
                    'engagement_rate': round(engagement, 4),
                    'post_count': max(1, base_posts),
                    'likes': int(base_posts * engagement * random.uniform(15, 60)),
                    'shares': int(base_posts * engagement * random.uniform(3, 20)),
                    'comments': int(base_posts * engagement * random.uniform(1, 12)),
                    'buzz_type': random.choice(['Social Media', 'News', 'Trailer', 'Reviews', 'Celebrity']),
                    'label': success,
                    'actual_budget': movie['budget'],
                    'actual_revenue': movie['revenue'],
                    'actual_rating': movie['rating'],
                    'actual_popularity': (movie['revenue'] / movie['budget']) * 10 if movie['budget'] > 0 else 0
                }
                
                social_data.append(social_record)
        
        return social_data

def main():
    """Main function to fetch real data"""
    print("🎬 Real Movie Data Fetcher Starting...")
    
    # Initialize fetcher
    fetcher = RealMovieDataFetcher()
    
    # Fetch real data
    real_data = fetcher.fetch_real_data(use_cache=True)
    
    # Save to the dashboard data file
    output_file = 'real_movie_dataset.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(real_data, f, indent=2)
        print(f"💾 Saved {len(real_data)} records to {output_file}")
        
        # Print summary statistics
        df = pd.DataFrame(real_data)
        print("\n📊 Dataset Summary:")
        print(f"   Total Records: {len(df)}")
        print(f"   Unique Movies: {df['movie_name'].nunique()}")
        print(f"   Success Distribution:")
        print(f"      {df['label'].value_counts().to_dict()}")
        print(f"   Average Engagement: {df['engagement_rate'].mean():.3f}")
        print(f"   Date Range: {df['days_before_release'].min()} to {df['days_before_release'].max()} days")
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")

if __name__ == "__main__":
    main()
