import time
from src.agents.base import BaseAgent
from src.schemas import Message

class SynthesizerAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("synthesizer", message_bus)

    def handle_message(self, message: Message):
        if message.msg_type == "synthesize_request":
            self.logger.info(f"Received synthesize request for {message.request_id}")
            start_time = time.time()
            
            topic = message.payload.get("topic")
            sub_queries = message.payload.get("sub_queries", [])
            all_sources = message.payload.get("sources", [])
            
            # Simulate synthesis
            time.sleep(0.5)
            
            sections = []
            for query in sub_queries:
                # Find sources related to this query based on title heuristic
                query_sources = [s for s in all_sources if query in s["title"]]
                source_ids = [s["source_id"] for s in query_sources]
                
                if source_ids:
                    sections.append({
                        "heading": f"Analysis of {query}",
                        "content": f"This section resolves conflicting information and synthesizes details about {query}. The consensus is that it is a vital aspect of {topic}.",
                        "citations": source_ids
                    })
            
            summary = f"An executive summary of {topic}. The research covers {len(sub_queries)} main aspects. The overall findings indicate a strong correlation between the sub-topics, synthesized from {len(all_sources)} sources."
            
            elapsed = time.time() - start_time
            
            self.send_message(
                request_id=message.request_id,
                recipient="supervisor",
                msg_type="synthesize_result",
                payload={
                    "summary": summary,
                    "sections": sections,
                    "synthesis_time": elapsed
                }
            )
