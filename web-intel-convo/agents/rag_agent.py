import logging
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from config import config
import os

logger = logging.getLogger("RAGAgent")

class ConversationalRAGAgent:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=config.db_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="web_intel_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        os.environ["GOOGLE_API_KEY"] = config.gemini_api_key
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=config.embeddings_model)
        
        self.llm = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0.1)
        
        # Session state memory: dict mapping session_id -> list of message dicts (role, content)
        self.sessions = {}

    def _get_history_text(self, session_id: str) -> str:
        history = self.sessions.get(session_id, [])
        if not history:
            return ""
        
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        # Token estimation: rough approximation (1 token ~= 4 chars)
        if len(history_text) > config.max_history_tokens * 4:
            logger.info(f"Summarizing conversation history for session {session_id}")
            summary = self._summarize_history(history_text)
            self.sessions[session_id] = [{"role": "system", "content": f"Previous conversation summary: {summary}"}]
            return self.sessions[session_id][0]["content"]
            
        return history_text

    def _summarize_history(self, history_text: str) -> str:
        prompt = PromptTemplate(
            input_variables=["history"],
            template="Summarize the following conversation history briefly while retaining key facts and coreferences.\n\nHistory:\n{history}\n\nSummary:"
        )
        try:
            chain = prompt | self.llm
            response = chain.invoke({"history": history_text})
            return response.content
        except Exception as e:
            logger.error(f"Failed to summarize history: {e}")
            return "History summary unavailable."

    def _retrieve_and_rerank(self, query: str) -> list[dict]:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            # Retrieve top_k directly since we removed local reranker to fix Windows path limits
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=config.top_k_retrieve
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        chunks = [{"text": doc, "metadata": meta} for doc, meta in zip(docs, metas)]

        # LLM-based Re-ranking
        logger.info("Performing LLM-based re-ranking on retrieved chunks...")
        for chunk in chunks:
            rerank_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Rate the relevance of the following context to the query on a scale of 0 to 10. Output only the number (e.g., 8).\n\nQuery: {query}\n\nContext: {context}\n\nScore:"
            )
            try:
                score_str = (rerank_prompt | self.llm).invoke({"query": query, "context": chunk["text"]}).content.strip()
                chunk["score"] = float(score_str)
            except Exception as e:
                logger.warning(f"Failed to score chunk: {e}")
                chunk["score"] = 0.0
                
        chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

        # Return top_k
        return chunks[:config.top_k_retrieve]

    def chat(self, session_id: str, query: str) -> tuple[str, list[dict]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        history_text = self._get_history_text(session_id)
        
        # Rewrite query if there is history (to resolve coreferences like "tell me more about that")
        search_query = query
        if history_text:
            rewrite_prompt = PromptTemplate(
                input_variables=["history", "query"],
                template="Given the following conversation history and the user's latest follow-up question, rewrite the follow-up question to be a standalone search query that contains all necessary context from the history.\n\nHistory:\n{history}\n\nFollow-up: {query}\n\nStandalone Query:"
            )
            try:
                rewrite_chain = rewrite_prompt | self.llm
                rewrite_resp = rewrite_chain.invoke({"history": history_text, "query": query})
                search_query = rewrite_resp.content.strip()
                logger.info(f"Rewrote query to: {search_query}")
            except Exception as e:
                logger.error(f"Failed to rewrite query: {e}")

        retrieved_chunks = self._retrieve_and_rerank(search_query)

        if not retrieved_chunks:
            msg = "I could not find any relevant information in the knowledge base regarding your query."
            self.sessions[session_id].append({"role": "user", "content": query})
            self.sessions[session_id].append({"role": "assistant", "content": msg})
            return msg, []

        context_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            source_id = f"[{i+1}]"
            context_text += f"Source {source_id} - URL: {chunk['metadata'].get('url', 'Unknown')}\nContent: {chunk['text']}\n\n"

        qa_prompt = PromptTemplate(
            input_variables=["context", "history", "query"],
            template="""You are a helpful and accurate AI assistant. Use the following retrieved context to answer the user's question. 
If the answer is not contained within the context, you must explicitly state that you don't have the information. Do not hallucinate.
When you use information from the context, include inline citations like [1], [2] at the end of the relevant sentence.

Conversation History:
{history}

Retrieved Context:
{context}

User Question: {query}
Answer:"""
        )

        try:
            qa_chain = qa_prompt | self.llm
            response = qa_chain.invoke({"context": context_text, "history": history_text, "query": query})
            answer = response.content
            
            if isinstance(answer, list):
                answer = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in answer])
            elif not isinstance(answer, str):
                answer = str(answer)
            
            self.sessions[session_id].append({"role": "user", "content": query})
            self.sessions[session_id].append({"role": "assistant", "content": answer})
            
            return answer, retrieved_chunks
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "An error occurred while generating the answer.", []
