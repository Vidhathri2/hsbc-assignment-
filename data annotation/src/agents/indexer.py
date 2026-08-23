from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever

from src.schema import Sample
from src.logger import get_logger

logger = get_logger("IndexerAgent")

class IndexerAgent:
    def __init__(self, client):
        self.client = client
        self.documents: List[Document] = []
        self.retriever = None
        
    def summarize_and_index(self, samples: List[Sample]):
        logger.info(f"Indexing {len(samples)} samples via LangChain Document storage...")
        
        for sample in samples:
            sample.summary = "Extracted from full text."
            
            # Create LangChain Document
            doc = Document(
                page_content=f"Source: {sample.source}\nLabel: {sample.label}\nFull Text: {sample.text}",
                metadata={"id": sample.id, "label": sample.label, "source": sample.source}
            )
            self.documents.append(doc)
            
        # Build LangChain TF-IDF Retriever
        if self.documents:
            self.retriever = TFIDFRetriever.from_documents(self.documents)
                
        logger.info(f"Successfully indexed {len(samples)} LangChain Documents.")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.retriever:
            return []
            
        self.retriever.k = top_k
        docs = self.retriever.invoke(query)
        
        return [doc.page_content for doc in docs]
