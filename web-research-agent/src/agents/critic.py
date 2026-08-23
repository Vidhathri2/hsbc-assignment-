import time
from src.agents.base import BaseAgent
from src.schemas import Message

class CriticAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("critic", message_bus)

    def handle_message(self, message: Message):
        if message.msg_type == "critique_request":
            self.logger.info(f"Received critique request for {message.request_id}")
            start_time = time.time()
            
            iteration = message.payload.get("iteration", 0)
            sections = message.payload.get("sections", [])
            
            # Simulate critique
            time.sleep(0.3)
            
            gaps = []
            bias_flags = []
            
            # Base confidence on iteration to force re-search loops (max 2)
            # If iteration == 0, mock low confidence to trigger one research loop randomly or deterministically
            if iteration == 0 and len(sections) < 5:
                confidence_score = 0.6
                gaps.append("Missing detailed coverage on sub-topic implications.")
            else:
                confidence_score = 0.85
                if iteration > 0:
                    bias_flags.append("Slight reliance on overlapping source material.")
            
            elapsed = time.time() - start_time
            
            self.send_message(
                request_id=message.request_id,
                recipient="supervisor",
                msg_type="critique_result",
                payload={
                    "confidence_score": confidence_score,
                    "gaps": gaps,
                    "bias_flags": bias_flags,
                    "critique_time": elapsed
                }
            )
