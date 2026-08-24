import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logger import get_logger

load_dotenv()

logger = get_logger("LLMSetup")

def get_genai_client() -> ChatGoogleGenerativeAI:
    logger.info("Initializing Gemini API LLM...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.0
    )
    return llm
