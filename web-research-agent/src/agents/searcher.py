import time
import uuid
from datetime import datetime
from src.agents.base import BaseAgent
from src.schemas import Message, Source
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

class SearcherAgent(BaseAgent):
    """
    Executes actual web searches using DuckDuckGo.
    """
    def __init__(self, message_bus):
        super().__init__("searcher", message_bus)
        
    def handle_message(self, message: Message):
        if message.msg_type == "search_request":
            self.logger.info(f"Received search request for {message.request_id}")
            start_time = time.time()
            
            sub_query = message.payload.get("sub_query")
            max_sources_per_query = message.payload.get("max_sources", 3)
            
            sources = []
            if DDGS:
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(sub_query, max_results=max_sources_per_query))
                        for res in results:
                            source_id = str(uuid.uuid4())
                            sources.append({
                                "source_id": source_id,
                                "url": res.get("href", ""),
                                "title": res.get("title", ""),
                                "snippet": res.get("body", ""),
                                "relevance_score": 1.0,
                                "scraped_at": datetime.utcnow().isoformat()
                            })
                except Exception as e:
                    self.logger.error(f"Search failed: {e}")
            
            elapsed = time.time() - start_time
            
            self.send_message(
                request_id=message.request_id,
                recipient="supervisor",
                msg_type="search_result",
                payload={
                    "sub_query": sub_query,
                    "sources": sources,
                    "search_time": elapsed,
                    "scrape_time": 0.0
                }
            )
