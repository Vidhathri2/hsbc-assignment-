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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

    import json
    
    # Reduce logging spam for interactive mode
    logging.getLogger().setLevel(logging.WARNING)
    
    print("\n" + "="*50)
    print("WEB RESEARCH AGENT - INTERACTIVE MODE")
    print("Type your topic to research, or 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        try:
            topic = input("Enter a research topic: ").strip()
            if topic.lower() in ['exit', 'quit']:
                break
            if not topic:
                continue
                
            req_id = str(uuid.uuid4())
            supervisor.submit_request(req_id, {
                "topic": topic,
                "depth": "shallow",
                "max_sources": 5,
                "output_format": "json"
            })
            
            print(f"[*] Researching '{topic}'... The agents are now working. Please wait.")
            
            # Wait for completion
            result_file = f"results/{req_id}.json"
            start_time = time.time()
            
            while not os.path.exists(result_file):
                time.sleep(1)
                if time.time() - start_time > 300: # 5 min timeout
                    print("[!] Request timed out!")
                    break
                    
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                print("\n" + "="*50)
                print("FINAL RESEARCH REPORT")
                print("="*50)
                print(f"TOPIC: {data.get('topic')}\n")
                print(f"SUMMARY: {data.get('summary')}\n")
                
                print("DETAILED SECTIONS:")
                for sec in data.get('sections', []):
                    print(f"\n--- {sec.get('heading')} ---")
                    print(sec.get('content'))
                    citations = sec.get('citations', [])
                    if citations:
                        print(f"Sources Cited: {len(citations)}")
                print("="*50 + "\n")
                
        except (KeyboardInterrupt, EOFError):
            break

    
    for agent in agents:
        agent.stop()
        
if __name__ == "__main__":
    main()
