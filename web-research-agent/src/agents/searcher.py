import time
import uuid
import hashlib
from datetime import datetime
from src.agents.base import BaseAgent
from src.schemas import Message, Source

class SearcherAgent(BaseAgent):
    """
    Executes web searches and handles scraping. 
    Implements a mock strategy to simulate 10,000 pre-crawled URLs.
    """
    def __init__(self, message_bus):
        super().__init__("searcher", message_bus)
        
    def handle_message(self, message: Message):
        if message.msg_type == "search_request":
            self.logger.info(f"Received search request for {message.request_id}")
            start_time = time.time()
            
            sub_query = message.payload.get("sub_query")
            max_sources_per_query = message.payload.get("max_sources", 5)
            
            # Apply rate limit jitter
            time.sleep(0.1)
            
            sources = []
            for i in range(max_sources_per_query):
                source_id = str(uuid.uuid4())
                url_hash = hashlib.md5(f"{sub_query}_{i}".encode()).hexdigest()
                sources.append({
                    "source_id": source_id,
                    "url": f"https://mock-dataset.com/article_{url_hash[:8]}",
                    "title": f"Article about {sub_query} part {i+1}",
                    "relevance_score": 0.5 + (0.5 * (1.0 - (i / max_sources_per_query))),
                    "scraped_at": datetime.utcnow().isoformat()
                })
            
            # Scrape time mock
            time.sleep(0.2)
            
            elapsed = time.time() - start_time
            
            self.send_message(
                request_id=message.request_id,
                recipient="supervisor",
                msg_type="search_result",
                payload={
                    "sub_query": sub_query,
                    "sources": sources,
                    "search_time": 0.1,
                    "scrape_time": elapsed - 0.1
                }
            )
