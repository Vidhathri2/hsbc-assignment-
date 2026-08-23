import logging
import json
from datetime import datetime
from config import config
from agents.search_tool import SearchTool
from agents.vector_db import VectorDBAgent
import schedule
import time
import os

logger = logging.getLogger("Pipeline")

class DailyIngestionPipeline:
    def __init__(self):
        self.search_tool = SearchTool()
        self.vector_db = VectorDBAgent()
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)

    def run_pipeline(self):
        """Executes the daily ingestion pipeline."""
        logger.info(f"Starting daily ingestion pipeline for topic: '{config.topic}'")
        start_time = time.time()
        
        try:
            # 1. Search and scrape
            documents = self.search_tool.run(config.topic)
            
            if not documents:
                logger.warning("No documents scraped today. Pipeline finished early.")
                report = {"status": "skipped", "reason": "No documents found."}
            else:
                # 2. Index into VectorDB
                report = self.vector_db.process_and_index(documents)
                report["status"] = "success"
                
            # 3. Purge old documents
            self.vector_db.purge_expired()
            
            # Save report
            report["execution_time_seconds"] = round(time.time() - start_time, 2)
            report["topic"] = config.topic
            
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_path = f"reports/ingestion_report_{date_str}.json"
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Pipeline completed successfully. Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            
            # Save failure report
            failure_report = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "topic": config.topic
            }
            date_str = datetime.now().strftime("%Y-%m-%d")
            report_path = f"reports/ingestion_report_{date_str}_failed.json"
            with open(report_path, "w") as f:
                json.dump(failure_report, f, indent=4)

    def start_scheduler(self):
        """Starts the CRON-style scheduler."""
        logger.info(f"Scheduling daily pipeline at {config.schedule_time}")
        schedule.every().day.at(config.schedule_time).do(self.run_pipeline)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    pipeline = DailyIngestionPipeline()
    # Run once immediately for demonstration/initialization
    pipeline.run_pipeline()
    # Start scheduler
    pipeline.start_scheduler()
