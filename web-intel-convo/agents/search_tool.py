import logging
import time
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import config
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SearchTool")

class SearchTool:
    def __init__(self):
        # We must explicitly pass the API key or set it in the environment
        os.environ["GOOGLE_API_KEY"] = config.gemini_api_key
        self.llm = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0.7)
        self.ddgs = DDGS()

    def generate_sub_queries(self, topic: str) -> list[str]:
        """Generates sub-queries by decomposing the topic."""
        logger.info(f"Generating sub-queries for topic: {topic}")
        prompt = PromptTemplate(
            input_variables=["topic", "count"],
            template="You are an expert web researcher. Decompose the following topic into {count} distinct search queries to find the most recent news and developments.\nTopic: {topic}\nOutput only the queries, one per line without numbering or bullet points."
        )
        try:
            chain = prompt | self.llm
            response = chain.invoke({"topic": topic, "count": config.sub_queries_count})
            # Handle if response.content is a list of blocks instead of a string
            content = response.content
            if isinstance(content, list):
                content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            elif not isinstance(content, str):
                content = str(content)
                
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            logger.info(f"Generated {len(queries)} sub-queries: {queries}")
            return queries[:config.sub_queries_count]
        except Exception as e:
            logger.error(f"Failed to generate sub-queries: {e}")
            return [topic] # Fallback to original topic

    def search_urls(self, queries: list[str]) -> list[dict]:
        """Searches for URLs and deduplicates them."""
        logger.info("Searching web for URLs...")
        all_results = []
        seen_urls = set()

        for query in queries:
            try:
                # Use DDGS for search (free, no API key)
                results = self.ddgs.text(query, max_results=10)
                if results:
                    for r in results:
                        url = r.get("href")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append({
                                "url": url,
                                "title": r.get("title", ""),
                                "snippet": r.get("body", "")
                            })
                            if len(all_results) >= config.max_sources_per_run:
                                break
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                time.sleep(2) # Rate limit handling
                
            if len(all_results) >= config.max_sources_per_run:
                break

        logger.info(f"Found {len(all_results)} unique URLs.")
        return all_results

    def fetch_content(self, url: str) -> str:
        """Fetches and parses text content from a URL."""
        try:
            response = requests.get(url, timeout=config.search_timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            return text.strip()
        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""

    def run(self, topic: str) -> list[dict]:
        """Runs the search agent pipeline."""
        queries = self.generate_sub_queries(topic)
        search_results = self.search_urls(queries)
        
        documents = []
        for result in search_results:
            content = self.fetch_content(result["url"])
            if content and len(content) > 100: # Filter out empty or very short pages
                documents.append({
                    "url": result["url"],
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "content": content
                })
                
        logger.info(f"Successfully scraped content from {len(documents)} URLs.")
        return documents
