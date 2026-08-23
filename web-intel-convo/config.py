import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class SystemConfig(BaseModel):
    # Topic Configuration
    topic: str = Field(default="Advancements in AI in the medical field")
    
    # Search Agent Configuration
    max_sources_per_run: int = Field(default=20)
    sub_queries_count: int = Field(default=5)
    search_timeout_seconds: int = Field(default=10)
    
    # Vector DB & Indexer Configuration
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    retention_days: int = Field(default=30)
    db_persist_directory: str = Field(default="./chroma_db")
    embeddings_model: str = Field(default="models/gemini-embedding-2") # Gemini Embeddings
    
    # Conversational RAG Configuration
    top_k_retrieve: int = Field(default=5)
    max_history_tokens: int = Field(default=2000)
    llm_model: str = Field(default="gemini-3.5-flash") # Switched to Gemini
    
    # Gemini API config
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    # Scheduling
    schedule_time: str = Field(default="08:00") # Time to run daily pipeline

config = SystemConfig()
