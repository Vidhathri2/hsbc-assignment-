import os
from langchain_ollama import ChatOllama
from src.logger import get_logger

logger = get_logger("LLMSetup")

def get_genai_client() -> ChatOllama:
    logger.info("Initializing Local Ollama LLM...")
    
    # Using local Ollama. The user has llama2 installed.
    llm = ChatOllama(
        model="llama2",
        temperature=0.0
    )
    return llm
