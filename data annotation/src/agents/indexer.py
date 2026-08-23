from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.schema import Sample
from src.logger import get_logger

logger = get_logger("IndexerAgent")

class IndexerAgent:
    def __init__(self, client):
        self.client = client
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
    def summarize_and_index(self, samples: List[Sample]):
        logger.info(f"Indexing {len(samples)} samples locally...")
        
        for sample in samples:
            # We skip the LLM summarization to avoid hitting Gemini API Rate Limits (429 errors).
            # The RAG search will just use the raw text, which is actually more accurate!
            sample.summary = "Extracted from full text."
            
            # Store document locally
            document_content = f"Source: {sample.source}\nLabel: {sample.label}\nFull Text: {sample.text}"
            self.documents.append(document_content)
            
        # Build TF-IDF index for ultra-fast local search
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
                
        logger.info(f"Successfully indexed {len(samples)} samples locally.")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.documents or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        # Only return if there is some non-zero similarity
        results = [self.documents[i] for i in top_indices if sims[i] > 0.0]
        return results
