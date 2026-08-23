import pytest
from unittest.mock import MagicMock
from src.schemas import Message
from src.agents.planner import PlannerAgent
from src.agents.searcher import SearcherAgent

def test_planner_agent():
    bus_mock = MagicMock()
    planner = PlannerAgent(bus_mock)
    
    msg = Message(
        request_id="test_req",
        sender="supervisor",
        recipient="planner",
        msg_type="plan_request",
        payload={"topic": "Test Topic", "depth": "shallow"}
    )
    
    planner.handle_message(msg)
    
    assert bus_mock.publish.called
    published_msg = bus_mock.publish.call_args[0][0]
    assert published_msg.msg_type == "plan_result"
    assert published_msg.recipient == "supervisor"
    assert len(published_msg.payload["sub_queries"]) == 3
    assert published_msg.payload["search_strategy"] == "breadth-first"

def test_searcher_agent():
    bus_mock = MagicMock()
    searcher = SearcherAgent(bus_mock)
    
    msg = Message(
        request_id="test_req",
        sender="supervisor",
        recipient="searcher",
        msg_type="search_request",
        payload={"sub_query": "Test Topic aspect 1", "max_sources": 5}
    )
    
    searcher.handle_message(msg)
    
    assert bus_mock.publish.called
    published_msg = bus_mock.publish.call_args[0][0]
    assert published_msg.msg_type == "search_result"
    assert len(published_msg.payload["sources"]) == 5
    assert published_msg.payload["sub_query"] == "Test Topic aspect 1"
