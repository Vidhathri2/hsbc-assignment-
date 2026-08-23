from src.logger import get_logger
from src.agents.indexer import IndexerAgent

logger = get_logger("ChatAgent")

class ChatAgent:
    def __init__(self, client, indexer: IndexerAgent):
        self.client = client
        self.indexer = indexer

    def chat(self, query: str) -> str:
        # 1. Retrieve similar documents using the local TF-IDF search
        retrieved_docs = self.indexer.search(query, top_k=3)
        
        if not retrieved_docs:
            return "I'm sorry, I couldn't find any relevant information in the knowledge base."
            
        # 2. Construct grounded prompt
        context = "\n\n---\n\n".join(retrieved_docs)
        prompt = (
            "You are a helpful knowledge assistant. Based strictly on the following context, answer the user's query.\n"
            "If the answer is not in the context, say 'I cannot answer this based on the available data.'\n"
            "Ensure you cite the source of your information.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )
        
        # 3. Generate response
        try:
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            return "Error communicating with LLM."
