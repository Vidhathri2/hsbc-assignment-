import uuid
import random
import feedparser
from typing import List
from src.schema import Sample

def get_live_news_dataset(num_samples: int = 40) -> List[Sample]:
    """Fetches real-time news from live RSS feeds to form the unlabelled pool."""
    rss_feeds = {
        "BBC News": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex"
    }
    
    samples = []
    for source_name, url in rss_feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:15]: # Take top 15 from each
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            # Clean up HTML tags if present (simple strip)
            import re
            summary = re.sub('<[^<]+>', '', summary)
            
            text = f"{title}. {summary}".strip()
            if len(text) > 20: # Ensure valid text
                samples.append(
                    Sample(
                        id=str(uuid.uuid4()), 
                        text=text,
                        source=source_name
                    )
                )
    
    # Shuffle and pick the requested number of real-time samples
    random.shuffle(samples)
    return samples[:num_samples]

# Alias to avoid breaking main.py
get_mock_dataset = get_live_news_dataset
