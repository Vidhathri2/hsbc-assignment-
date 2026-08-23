import time
import os
import json
import logging
import threading
from typing import Dict, Any
from src.agents.base import BaseAgent
from src.schemas import Message, ResearchOutput, Metadata, Source, Section, Critique

class SupervisorAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("supervisor", message_bus)
        self.active_requests = {}
        self.lock = threading.RLock()
        
        # Start a timeout monitor
        self.monitor_thread = threading.Thread(target=self._monitor_timeouts, daemon=True)
        self.monitor_thread.start()
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)

    def submit_request(self, request_id: str, payload: dict):
        with self.lock:
            self.active_requests[request_id] = {
                "request_id": request_id,
                "payload": payload,
                "state": "planning",
                "start_time": time.time(),
                "plan": {},
                "sources": [],
                "pending_searches": 0,
                "report": {},
                "iteration": 0,
                "metadata": {
                    "total_urls_visited": 0,
                    "agent_interactions": 0,
                    "planning_time": 0.0,
                    "search_time": 0.0,
                    "scrape_time": 0.0,
                    "synthesis_time": 0.0,
                    "critique_time": 0.0,
                    "research_time": 0.0
                }
            }
        
        self._increment_interaction(request_id)
        self.send_message(request_id, "planner", "plan_request", payload)

    def _increment_interaction(self, request_id: str):
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["metadata"]["agent_interactions"] += 1

    def handle_message(self, message: Message):
        req_id = message.request_id
        
        with self.lock:
            if req_id not in self.active_requests:
                return # Request might have timed out or completed
            req_data = self.active_requests[req_id]

        self._increment_interaction(req_id)

        if message.msg_type == "plan_result":
            self._handle_plan_result(req_id, req_data, message.payload)
        elif message.msg_type == "search_result":
            self._handle_search_result(req_id, req_data, message.payload)
        elif message.msg_type == "synthesize_result":
            self._handle_synthesize_result(req_id, req_data, message.payload)
        elif message.msg_type == "critique_result":
            self._handle_critique_result(req_id, req_data, message.payload)

    def _handle_plan_result(self, req_id: str, req_data: dict, payload: dict):
        with self.lock:
            req_data["plan"] = payload
            req_data["metadata"]["planning_time"] += payload.get("planning_time", 0.0)
            req_data["state"] = "searching"
            
            sub_queries = payload.get("sub_queries", [])
            req_data["pending_searches"] = len(sub_queries)
            
        for query in sub_queries:
            self.send_message(req_id, "searcher", "search_request", {
                "sub_query": query,
                "max_sources": req_data["payload"].get("max_sources", 5)
            })

    def _handle_search_result(self, req_id: str, req_data: dict, payload: dict):
        with self.lock:
            sources = payload.get("sources", [])
            req_data["sources"].extend(sources)
            req_data["metadata"]["search_time"] += payload.get("search_time", 0.0)
            req_data["metadata"]["scrape_time"] += payload.get("scrape_time", 0.0)
            req_data["metadata"]["total_urls_visited"] += len(sources)
            
            req_data["pending_searches"] -= 1
            if req_data["pending_searches"] <= 0:
                req_data["state"] = "synthesizing"
                self.send_message(req_id, "synthesizer", "synthesize_request", {
                    "topic": req_data["payload"]["topic"],
                    "sub_queries": req_data["plan"].get("sub_queries", []),
                    "sources": req_data["sources"]
                })

    def _handle_synthesize_result(self, req_id: str, req_data: dict, payload: dict):
        with self.lock:
            req_data["report"] = payload
            req_data["metadata"]["synthesis_time"] += payload.get("synthesis_time", 0.0)
            req_data["state"] = "critiquing"
            
        self.send_message(req_id, "critic", "critique_request", {
            "iteration": req_data["iteration"],
            "sections": payload.get("sections", [])
        })

    def _handle_critique_result(self, req_id: str, req_data: dict, payload: dict):
        with self.lock:
            req_data["metadata"]["critique_time"] += payload.get("critique_time", 0.0)
            confidence = payload.get("confidence_score", 0.0)
            
            if confidence < 0.7 and req_data["iteration"] < 2:
                # Trigger re-search
                req_data["iteration"] += 1
                req_data["state"] = "searching"
                req_data["pending_searches"] = 1
                # Ask for one more specific query based on gaps
                gaps = payload.get("gaps", [])
                gap_query = gaps[0] if gaps else f"More on {req_data['payload']['topic']}"
                req_data["plan"]["sub_queries"].append(gap_query)
                self.send_message(req_id, "searcher", "search_request", {
                    "sub_query": gap_query,
                    "max_sources": 5
                })
            else:
                self._finalize_request(req_id, req_data, payload)

    def _finalize_request(self, req_id: str, req_data: dict, critique_payload: dict):
        # Finalize and write to output
        req_data["state"] = "completed"
        wall_clock = time.time() - req_data["start_time"]
        req_data["metadata"]["wall_clock_seconds"] = wall_clock
        req_data["metadata"]["research_time"] = sum([
            req_data["metadata"]["planning_time"],
            req_data["metadata"]["search_time"],
            req_data["metadata"]["scrape_time"],
            req_data["metadata"]["synthesis_time"],
            req_data["metadata"]["critique_time"]
        ])
        
        output = ResearchOutput(
            report_id=req_id,
            topic=req_data["payload"]["topic"],
            summary=req_data["report"].get("summary", ""),
            sections=[Section(**s) for s in req_data["report"].get("sections", [])],
            sources=[Source(**s) for s in req_data["sources"]],
            critique=Critique(**critique_payload),
            metadata=Metadata(**req_data["metadata"])
        )
        
        # Deduplicate sources based on source_id in case of overlapping searches
        unique_sources = {s.source_id: s for s in output.sources}
        output.sources = list(unique_sources.values())
        
        with open(f"results/{req_id}.json", "w") as f:
            f.write(output.json(indent=2))
            
        self.logger.info(f"Completed request {req_id} in {wall_clock:.2f}s")
        with self.lock:
            del self.active_requests[req_id]

    def _monitor_timeouts(self):
        while True:
            time.sleep(10)
            now = time.time()
            timeout_ids = []
            with self.lock:
                for req_id, req_data in self.active_requests.items():
                    if now - req_data["start_time"] > 300: # 5 minutes global timeout
                        timeout_ids.append(req_id)
                for req_id in timeout_ids:
                    self.logger.warning(f"Request {req_id} timed out.")
                    del self.active_requests[req_id]
