import os
import time
import pytest
from src.message_bus import MessageBus
from src.supervisor import SupervisorAgent
from src.agents.planner import PlannerAgent
from src.agents.searcher import SearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.critic import CriticAgent

@pytest.fixture(scope="module")
def setup_agents():
    host = os.getenv("REDIS_HOST", "localhost")
    bus = MessageBus(host=host)
    
    supervisor = SupervisorAgent(bus)
    planner = PlannerAgent(bus)
    searcher = SearcherAgent(bus)
    synthesizer = SynthesizerAgent(bus)
    critic = CriticAgent(bus)
    
    agents = [supervisor, planner, searcher, synthesizer, critic]
    for agent in agents:
        agent.start()
        
    yield supervisor
    
    for agent in agents:
        agent.stop()
    bus.close()

def test_full_pipeline(setup_agents):
    supervisor = setup_agents
    req_id = "test_int_1"
    
    supervisor.submit_request(req_id, {
        "topic": "Quantum Computing",
        "depth": "shallow",
        "max_sources": 5,
        "output_format": "json"
    })
    
    # Wait for processing
    timeout = 10
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(f"results/{req_id}.json"):
            break
        time.sleep(0.5)
        
    assert os.path.exists(f"results/{req_id}.json")
    
    # Basic validation
    import json
    with open(f"results/{req_id}.json", "r") as f:
        data = json.load(f)
        assert data["report_id"] == req_id
        assert len(data["sections"]) > 0
        assert len(data["sources"]) > 0
        assert "confidence_score" in data["critique"]
