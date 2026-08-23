import os
import time
import uuid
import logging
from src.message_bus import MessageBus
from src.supervisor import SupervisorAgent
from src.agents.planner import PlannerAgent
from src.agents.searcher import SearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.critic import CriticAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    bus = MessageBus(host=os.getenv("REDIS_HOST", "localhost"))
    
    # Initialize and start agents
    # To optimize for throughput (4 cores, processing 100 topics in 10 mins),
    # we utilize thread concurrency and Redis Stream consumer groups.
    # Multiple workers of the same type naturally load-balance requests.
    supervisor = SupervisorAgent(bus)
    planner = PlannerAgent(bus)
    searcher_1 = SearcherAgent(bus)
    searcher_2 = SearcherAgent(bus)  # Additional searcher for concurrency
    synthesizer = SynthesizerAgent(bus)
    critic = CriticAgent(bus)

    agents = [supervisor, planner, searcher_1, searcher_2, synthesizer, critic]
    for agent in agents:
        agent.start()

    logging.info("All agents started. Submitting 100 research topics...")
    start_time = time.time()
    
    request_ids = []
    for i in range(100):
        req_id = str(uuid.uuid4())
        request_ids.append(req_id)
        supervisor.submit_request(req_id, {
            "topic": f"Machine Learning in Healthcare {i}",
            "depth": "shallow",
            "max_sources": 5,
            "output_format": "json"
        })
        time.sleep(0.01) # Stagger submission slightly

    # Wait for completion
    while True:
        completed = len(os.listdir("results")) if os.path.exists("results") else 0
        logging.info(f"Progress: {completed}/100 completed")
        if completed >= 100:
            break
        time.sleep(2)
        if time.time() - start_time > 600:
            logging.error("Global timeout of 10 minutes reached!")
            break

    total_time = time.time() - start_time
    logging.info(f"Finished 100 topics in {total_time:.2f} seconds.")
    
    for agent in agents:
        agent.stop()
        
if __name__ == "__main__":
    main()
