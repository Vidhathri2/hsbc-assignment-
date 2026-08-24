import time
import uuid
import os
from typing import List, Dict, Any
from src.agents.base import BaseAgent
from src.schemas import Message

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

class PlannerAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("planner", message_bus)
        if ChatGoogleGenerativeAI and os.getenv("GEMINI_API_KEY"):
            self.model = ChatGoogleGenerativeAI(model="gemini-3.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.model = None

    def handle_message(self, message: Message):
        if message.msg_type == "plan_request":
            self.logger.info(f"Received plan request for {message.request_id}")
            start_time = time.time()
            
            topic = message.payload.get("topic")
            depth = message.payload.get("depth", "shallow")
            
            num_queries = 3 if depth == "shallow" else (5 if depth == "moderate" else 8)
            
            sub_queries = []
            if self.model:
                try:
                    prompt = f"I am researching: '{topic}'. Give me exactly {num_queries} specific search queries I should run to gather information on this topic. Output ONLY the queries, one per line, with no bullet points or numbers."
                    response = self.model.invoke(prompt)
                    sub_queries = [line.strip() for line in response.content.strip().split('\n') if line.strip()][:num_queries]
                except Exception as e:
                    self.logger.error(f"Planner LLM Error: {e}")
            
            if not sub_queries:
                sub_queries = [f"{topic} overview", f"{topic} benefits", f"{topic} examples"]
            
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
