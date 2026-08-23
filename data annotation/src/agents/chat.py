from langchain_core.prompts import PromptTemplate
from src.logger import get_logger
from src.agents.indexer import IndexerAgent

logger = get_logger("ChatAgent")

class ChatAgent:
    def __init__(self, llm, indexer: IndexerAgent):
        self.llm = llm
        self.indexer = indexer
        
        self.prompt_template = PromptTemplate.from_template(
            "You are a helpful knowledge assistant. Based strictly on the following context, answer the user's query.\n"
            "If the answer is not in the context, say 'I cannot answer this based on the available data.'\n"
            "Ensure you cite the source of your information.\n\n"
            "Context:\n{context}\n\n"
            "User Query: {query}\n\n"
            "Answer:"
        )

    def chat(self, query: str) -> str:
        # 1. Retrieve similar documents using the local TF-IDF search
        retrieved_docs = self.indexer.search(query, top_k=3)
        
        if not retrieved_docs:
            return "I'm sorry, I couldn't find any relevant information in the knowledge base."
            
        # 2. Construct grounded prompt
        context = "\n\n---\n\n".join(retrieved_docs)
        
        # 3. Generate response via LangChain Core
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "context": context,
                "query": query
            })
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            return "Error communicating with LLM."
