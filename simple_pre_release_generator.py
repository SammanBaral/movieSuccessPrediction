"""
Simplified Pre-Release Movie Data Generator
No external dependencies required - generates realistic training data
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class SimplePreReleaseDataGenerator:
    def __init__(self):
        self.buzz_templates = {
            'announcement': [
                "{movie} officially announced! Can't wait! #ComingSoon #Excited #MovieNews",
                "OMG! {movie} is happening! This is going to be amazing! #Hype #CantWait",
                "Just heard about {movie}! {director} directing! Perfect! #MovieBuzz #Excited"
            ],
            'casting_news': [
                "{actor} confirmed for {movie}! Perfect casting! #PerfectCast #Excited",
                "The cast for {movie} looks incredible! {actor} is perfect! #DreamCast #Hype",
                "So excited to see {actor} in {movie}! This will be epic! #CastingNews #Amazing"
            ],
            'first_trailer': [
                "HOLY! The {movie} trailer is INSANE! #TrailerDrop #MustWatch #Epic",
                "Just watched the {movie} trailer 10 times! This will be HUGE! #Trailer #Amazing",
                "The {movie} trailer gave me chills! Can't wait! #TrailerReaction #Excited"
            ],
            'final_trailer': [
                "Final {movie} trailer! I'm not ready for this! #FinalTrailer #Emotional #MustSee",
                "This final {movie} trailer has me in TEARS! Going to be incredible! #Amazing",
                "If the final trailer is this good, {movie} will be a masterpiece! #Epic"
            ],
            'premiere_buzz': [
                "Premiere reactions for {movie} are incredible! #Premiere #EarlyReactions #Hype",
                "Everyone at the {movie} premiere is going crazy! #PremiereNight #Epic",
                "The {movie} premiere reactions have me SO excited! #Premiere #CantWait"
            ],
            'early_reactions': [
                "Early reactions to {movie} are PHENOMENAL! Must see! #EarlyReactions #MustSee",
                "Critics are raving about {movie}! This is special! #CriticsChoice #Raves",
                "Early word on {movie} is amazing! Can't wait! #EarlyWord #Phenomenal"
            ]
        }
        
        self.movie_database = {
            # Successful Movies (Hits)
            "Avengers Endgame": {
                "year": "2019", "genre": "Action", "budget": 356, "director": "Russo Brothers",
                "actors": ["Robert Downey Jr", "Chris Evans", "Scarlett Johansson"],
                "studio": "Marvel Studios", "franchise": True, "sequel": True
            },
            "Top Gun Maverick": {
                "year": "2022", "genre": "Action", "budget": 170, "director": "Joseph Kosinski",
                "actors": ["Tom Cruise", "Miles Teller", "Jennifer Connelly"],
                "studio": "Paramount", "franchise": True, "sequel": True
            },
            "Barbie": {
                "year": "2023", "genre": "Comedy", "budget": 145, "director": "Greta Gerwig",
                "actors": ["Margot Robbie", "Ryan Gosling", "Will Ferrell"],
                "studio": "Warner Bros", "franchise": False, "sequel": False
            },
            "Oppenheimer": {
                "year": "2023", "genre": "Drama", "budget": 100, "director": "Christopher Nolan",
                "actors": ["Cillian Murphy", "Emily Blunt", "Robert Downey Jr"],
                "studio": "Universal", "franchise": False, "sequel": False
            },
            "Spider-Man No Way Home": {
                "year": "2021", "genre": "Action", "budget": 200, "director": "Jon Watts",
                "actors": ["Tom Holland", "Zendaya", "Benedict Cumberbatch"],
                "studio": "Sony Pictures", "franchise": True, "sequel": True
            },
            
            # Average Performing Movies
            "Dune": {
                "year": "2021", "genre": "Sci-Fi", "budget": 165, "director": "Denis Villeneuve",
                "actors": ["Timothée Chalamet", "Rebecca Ferguson", "Oscar Isaac"],
                "studio": "Warner Bros", "franchise": True, "sequel": False
            },
            "Black Widow": {
                "year": "2021", "genre": "Action", "budget": 200, "director": "Cate Shortland",
                "actors": ["Scarlett Johansson", "Florence Pugh", "David Harbour"],
                "studio": "Marvel Studios", "franchise": True, "sequel": False
            },
            "Eternals": {
                "year": "2021", "genre": "Action", "budget": 200, "director": "Chloé Zhao",
                "actors": ["Gemma Chan", "Richard Madden", "Angelina Jolie"],
                "studio": "Marvel Studios", "franchise": True, "sequel": False
            },
            
            # Underperforming Movies (Flops)
            "The Flash": {
                "year": "2023", "genre": "Action", "budget": 220, "director": "Andy Muschietti",
                "actors": ["Ezra Miller", "Michael Keaton", "Sasha Calle"],
                "studio": "Warner Bros", "franchise": True, "sequel": False
            },
            "The Marvels": {
                "year": "2023", "genre": "Action", "budget": 220, "director": "Nia DaCosta",
                "actors": ["Brie Larson", "Teyonah Parris", "Iman Vellani"],
                "studio": "Marvel Studios", "franchise": True, "sequel": True
            },
            "Dark Phoenix": {
                "year": "2019", "genre": "Action", "budget": 200, "director": "Simon Kinberg",
                "actors": ["Sophie Turner", "James McAvoy", "Michael Fassbender"],
                "studio": "20th Century Fox", "franchise": True, "sequel": True
            },
            "Cats": {
                "year": "2019", "genre": "Musical", "budget": 95, "director": "Tom Hooper",
                "actors": ["James Corden", "Judi Dench", "Jason Derulo"],
                "studio": "Universal", "franchise": False, "sequel": False
            }
        }
        
        # Map movies to their actual success levels
        self.success_mapping = {
            # Hits
            "Avengers Endgame": "Hit",
            "Top Gun Maverick": "Hit", 
            "Barbie": "Hit",
            "Oppenheimer": "Hit",
            "Spider-Man No Way Home": "Hit",
            
            # Average
            "Dune": "Average",
            "Black Widow": "Average",
            "Eternals": "Average",
            
            # Flops
            "The Flash": "Flop",
            "The Marvels": "Flop", 
            "Dark Phoenix": "Flop",
            "Cats": "Flop"
        }
    
    def generate_buzz_for_movie(self, movie_title: str, success_level: str, samples_per_movie: int = 200) -> List[Dict]:
        """Generate pre-release buzz for a specific movie"""
        
        if movie_title not in self.movie_database:
            return []
            
        movie_info = self.movie_database[movie_title]
        buzz_data = []
        
        # Define buzz distribution based on eventual success
        if success_level == "Hit":
            buzz_distribution = {
                'announcement': 20, 'casting_news': 25, 'first_trailer': 40,
                'final_trailer': 50, 'premiere_buzz': 35, 'early_reactions': 30
            }
            engagement_multiplier = 2.0
        elif success_level == "Average":
            buzz_distribution = {
                'announcement': 15, 'casting_news': 20, 'first_trailer': 30,
                'final_trailer': 35, 'premiere_buzz': 25, 'early_reactions': 20
            }
            engagement_multiplier = 1.0
        else:  # Flop
            buzz_distribution = {
                'announcement': 10, 'casting_news': 15, 'first_trailer': 25,
                'final_trailer': 20, 'premiere_buzz': 15, 'early_reactions': 10
            }
            engagement_multiplier = 0.5
        
        # Generate release date
        year = int(movie_info['year'])
        release_date = datetime(year, random.randint(3, 11), random.randint(1, 28))
        
        # Generate buzz for each type
        for buzz_type, base_count in buzz_distribution.items():
            count = int(base_count * random.uniform(0.8, 1.2))
            
            for _ in range(count):
                # Timing based on buzz type
                timing_ranges = {
                    'announcement': (180, 365),
                    'casting_news': (120, 300), 
                    'first_trailer': (90, 180),
                    'final_trailer': (30, 60),
                    'premiere_buzz': (1, 14),
                    'early_reactions': (1, 7)
                }
                
                min_days, max_days = timing_ranges[buzz_type]
                days_before = random.randint(min_days, max_days)
                post_date = release_date - timedelta(days=days_before)
                
                # Generate text
                template = random.choice(self.buzz_templates[buzz_type])
                text = template.format(
                    movie=movie_title,
                    director=movie_info['director'],
                    actor=random.choice(movie_info['actors'])
                )
                
                # Generate engagement
                base_engagement = {
                    'announcement': {'likes': 150, 'shares': 30, 'comments': 20},
                    'casting_news': {'likes': 200, 'shares': 40, 'comments': 25},
                    'first_trailer': {'likes': 800, 'shares': 160, 'comments': 100},
                    'final_trailer': {'likes': 1200, 'shares': 240, 'comments': 150},
                    'premiere_buzz': {'likes': 500, 'shares': 100, 'comments': 60},
                    'early_reactions': {'likes': 600, 'shares': 120, 'comments': 80}
                }
                
                base = base_engagement[buzz_type]
                likes = int(base['likes'] * engagement_multiplier * random.uniform(0.5, 1.5))
                shares = int(base['shares'] * engagement_multiplier * random.uniform(0.5, 1.5))
                comments = int(base['comments'] * engagement_multiplier * random.uniform(0.5, 1.5))
                
                # Generate hashtags
                hashtag_pools = {
                    'announcement': ['#MovieNews', '#ComingSoon', '#Excited'],
                    'casting_news': ['#PerfectCast', '#CastingNews', '#Excited'],
                    'first_trailer': ['#TrailerDrop', '#MustWatch', '#Epic'],
                    'final_trailer': ['#FinalTrailer', '#MustSee', '#Amazing'],
                    'premiere_buzz': ['#Premiere', '#EarlyReactions', '#Hype'],
                    'early_reactions': ['#EarlyReactions', '#MustSee', '#Phenomenal']
                }
                
                hashtags = random.sample(hashtag_pools[buzz_type], 2)
                hashtags.append(f"#{movie_info['genre']}")
                
                buzz_data.append({
                    'text': text,
                    'hashtags': hashtags,
                    'likes': likes,
                    'shares': shares,
                    'comments': comments,
                    'source': f'pre_release_{buzz_type}',
                    'timestamp': post_date.isoformat(),
                    'movie_name': movie_title,
                    'label': success_level,
                    'buzz_type': buzz_type,
                    'days_before_release': days_before,
                    'movie_budget': movie_info['budget'],
                    'genre': movie_info['genre'],
                    'is_franchise': movie_info['franchise'],
                    'is_sequel': movie_info['sequel']
                })
        
        return buzz_data
    
    def generate_complete_dataset(self) -> List[Dict]:
        """Generate complete pre-release dataset"""
        all_data = []
        
        for movie_title, success_level in self.success_mapping.items():
            print(f"Generating data for {movie_title} ({success_level})...")
            movie_data = self.generate_buzz_for_movie(movie_title, success_level, 200)
            all_data.extend(movie_data)
            print(f"  Generated {len(movie_data)} samples")
        
        return all_data
    
    def save_data(self, data: List[Dict], filename: str):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Data saved to {filename}")

def main():
    """Generate the pre-release training dataset"""
    print("🎬 Pre-Release Movie Success Prediction Dataset Generator")
    print("=" * 60)
    print("🎯 FOCUS: Generate data from BEFORE movie release")
    print("📊 PURPOSE: Train model to predict success using pre-release buzz")
    print()
    
    generator = SimplePreReleaseDataGenerator()
    
    # Generate the complete dataset
    print("🔄 Generating pre-release buzz data...")
    all_data = generator.generate_complete_dataset()
    
    if all_data:
        # Save the dataset
        filename = "pre_release_movie_dataset.json"
        generator.save_data(all_data, filename)
        
        # Analysis
        print(f"\n📊 DATASET SUMMARY")
        print("=" * 40)
        print(f"Total samples: {len(all_data):,}")
        
        # Count by success level
        success_counts = {}
        buzz_counts = {}
        for item in all_data:
            success_counts[item['label']] = success_counts.get(item['label'], 0) + 1
            buzz_counts[item['buzz_type']] = buzz_counts.get(item['buzz_type'], 0) + 1
        
        print(f"\n🎯 Success Distribution:")
        for label, count in success_counts.items():
            percentage = count / len(all_data) * 100
            print(f"   {label}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n📅 Buzz Type Distribution:")
        for buzz_type, count in buzz_counts.items():
            print(f"   {buzz_type}: {count:,}")
        
        # Sample data
        print(f"\n📝 Sample Pre-Release Posts:")
        import random as rand
        for i, item in enumerate(rand.sample(all_data, 3)):
            print(f"   {i+1}. [{item['buzz_type']}] {item['text']}")
            print(f"      Movie: {item['movie_name']} | {item['days_before_release']} days before release")
            print(f"      Engagement: {item['likes']} likes | Actual outcome: {item['label']}")
            print()
        
        # Create test subset
        test_sample = rand.sample(all_data, min(100, len(all_data)))
        test_filename = "pre_release_test_sample.json"
        generator.save_data(test_sample, test_filename)
        
        print(f"💾 Files created:")
        print(f"   📁 {filename} - Full dataset ({len(all_data):,} samples)")
        print(f"   📁 {test_filename} - Test subset (100 samples)")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Train model: python enhanced_ml_pipeline.py {filename}")
        print(f"2. Use model to predict NEW movies using their pre-release buzz")
        print(f"3. Validate predictions against actual box office results")
        
        print(f"\n🎓 KEY INSIGHT:")
        print(f"This dataset simulates real-world scenario where you:")
        print(f"- Monitor social media buzz BEFORE movie release")
        print(f"- Use engagement patterns to predict success")
        print(f"- Help studios make informed decisions")
        
        return True
    else:
        print("❌ Failed to generate dataset!")
        return False

if __name__ == "__main__":
    main()
