import argparse
import sys
from pipeline import DailyIngestionPipeline
from chat import ChatCLI
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Web Intelligence RAG System")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["pipeline", "chat"], 
        required=True, 
        help="Run mode: 'pipeline' for daily ingestion, 'chat' for multi-turn conversational interface."
    )
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        pipeline = DailyIngestionPipeline()
        # Run immediately once, then start scheduler
        pipeline.run_pipeline()
        pipeline.start_scheduler()
    elif args.mode == "chat":
        # Decrease logging level for chat so it doesn't pollute standard output
        logging.getLogger().setLevel(logging.WARNING)
        ChatCLI().cmdloop()
    else:
        print("Invalid mode. Use --mode pipeline or --mode chat")
        sys.exit(1)

if __name__ == "__main__":
    main()
