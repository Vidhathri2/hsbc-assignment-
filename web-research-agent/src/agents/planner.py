import time
import uuid
from typing import List, Dict, Any
from src.agents.base import BaseAgent
from src.schemas import Message

class PlannerAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("planner", message_bus)

    def handle_message(self, message: Message):
        if message.msg_type == "plan_request":
            self.logger.info(f"Received plan request for {message.request_id}")
            start_time = time.time()
            
            topic = message.payload.get("topic")
            depth = message.payload.get("depth")
            
            # Decompose into 3-8 sub-queries based on depth
            num_queries = 3 if depth == "shallow" else (5 if depth == "moderate" else 8)
            sub_queries = [f"{topic} aspect {i+1}" for i in range(num_queries)]
            
            strategy = "breadth-first" if depth == "shallow" else "iterative_deepening"
            
            elapsed = time.time() - start_time
            
            self.send_message(
                request_id=message.request_id,
                recipient="supervisor",
                msg_type="plan_result",
                payload={
                    "sub_queries": sub_queries,
                    "search_strategy": strategy,
                    "planning_time": elapsed
                }
            )
