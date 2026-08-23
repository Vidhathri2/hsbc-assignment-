import os
from dotenv import load_dotenv

from src.llm import get_genai_client
from src.agents.annotator import AnnotatorAgent
from src.agents.quality_assessor import QualityAssessorAgent
from src.agents.trainer import TrainerAgent
from src.coordinator import Coordinator
from src.dataset import get_mock_dataset
from src.logger import get_logger

logger = get_logger("Main")

def main():
    try:
        # 1. Initialize LangChain LLM Client (Ollama)
        llm_client = get_genai_client()
        logger.info("LangChain Local Ollama client initialized successfully.")
    except Exception as e:
        logger.error(f"Initialization Error: {e}")
        return

    # 2. Fetch Unlabelled Data Pool
    unlabelled_data = get_mock_dataset()
    if not unlabelled_data:
        logger.error("Failed to fetch dataset. Exiting.")
        return

    # 3. Initialize Agentic Workers
    # Token budget for Annotator
    annotator = AnnotatorAgent(llm=llm_client, token_budget=10000)
    
    # QA Agent with confidence threshold 0.8
    qa_agent = QualityAssessorAgent(llm=llm_client, confidence_threshold=0.8)
    
    # Trainer Agent targeting 85% accuracy
    trainer = TrainerAgent(target_accuracy=0.85)

    # 4. Initialize the Hierarchical Multi-Agent Supervisor (Coordinator)
    coordinator = Coordinator(annotator, qa_agent, trainer)

    # Load initial unlabelled dataset
    coordinator.load_data(unlabelled_data)

    # Run the Hierarchical LangGraph Pipeline
    logger.info("Starting Hierarchical Multi-Agent Data Annotation Pipeline...")
    best_model, labelled_data = coordinator.run_pipeline(batch_size=2, max_iterations=10)

    logger.info(f"Pipeline finished. Total Labelled Samples: {len(labelled_data)}")
    
    if best_model:
        logger.info("A model was successfully trained.")
        
        # Grading Criteria 1: Index and Chat
        from src.agents.indexer import IndexerAgent
        from src.agents.chat import ChatAgent
        
        indexer = IndexerAgent(client=llm_client)
        indexer.summarize_and_index(labelled_data)
        
        chat_agent = ChatAgent(llm=llm_client, indexer=indexer)
        
        logger.info("\n--- Commencing Chat Test ---")
        test_queries = [
            "What happened with the stock market or tech companies recently?",
            "Any news related to sports or Olympics?"
        ]
        
        for q in test_queries:
            logger.info(f"User: {q}")
            response = chat_agent.chat(q)
            logger.info(f"Agent: {response}\n")
            
if __name__ == "__main__":
    main()
