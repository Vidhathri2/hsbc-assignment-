import time
import os
from src.agents.base import BaseAgent
from src.schemas import Message

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

class SynthesizerAgent(BaseAgent):
    def __init__(self, message_bus):
        super().__init__("synthesizer", message_bus)
        if ChatGoogleGenerativeAI and os.getenv("GEMINI_API_KEY"):
            self.model = ChatGoogleGenerativeAI(model="gemini-3.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.model = None

    def handle_message(self, message: Message):
        if message.msg_type == "synthesize_request":
            self.logger.info(f"Received synthesize request for {message.request_id}")
            start_time = time.time()
            
            topic = message.payload.get("topic")
            sub_queries = message.payload.get("sub_queries", [])
            all_sources = message.payload.get("sources", [])
            
            sections = []
            
            if self.model:
                try:
                    context = "\n".join([f"URL: {s.get('url')}\nTitle: {s.get('title')}\nSnippet: {s.get('snippet')}" for s in all_sources])
                    
                    # Consolidate into a single LLM call to avoid Rate Limit (429) errors!
                    prompt = f"""Topic: {topic}
Sub-topics: {', '.join(sub_queries)}

Context:
{context}

Please provide:
1. A 2-sentence executive summary of the topic.
2. A detailed paragraph for each of the sub-topics, synthesizing the provided context.

Format your response exactly like this:
SUMMARY: <your 2 sentence summary>
---
<Sub-topic 1>: <your paragraph>
---
<Sub-topic 2>: <your paragraph>
"""
                    response = self.model.invoke(prompt)
                    raw_text = response.content.strip()
                    
                    # Parse the single response
                    parts = raw_text.split("---")
                    summary = parts[0].replace("SUMMARY:", "").strip() if len(parts) > 0 else raw_text
                    
                    for i, part in enumerate(parts[1:]):
                        if ":" in part:
                            heading, content = part.split(":", 1)
                        else:
                            heading = f"Section {i+1}"
                            content = part
                            
                        sections.append({
                            "heading": heading.strip(),
                            "content": content.strip(),
                            "citations": [s.get("url") for s in all_sources[:2]]
                        })
                except Exception as e:
                    self.logger.error(f"LLM Error: {e}")
                    summary = f"Synthesis failed due to API error: {e}"
            else:
                summary = "Generative AI is not configured."
                
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
