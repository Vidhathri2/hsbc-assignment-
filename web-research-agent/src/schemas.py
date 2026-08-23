import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=200)
    depth: Literal["shallow", "moderate", "deep"]
    max_sources: int = Field(..., ge=5, le=50)
    output_format: Literal["markdown", "pdf", "json"]

class Section(BaseModel):
    heading: str
    content: str
    citations: List[str]

class Source(BaseModel):
    source_id: str
    url: str
    title: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    scraped_at: str

class Critique(BaseModel):
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    gaps: List[str]
    bias_flags: List[str]

class Metadata(BaseModel):
    total_urls_visited: int
    agent_interactions: int
    wall_clock_seconds: float
    planning_time: float = 0.0
    search_time: float = 0.0
    scrape_time: float = 0.0
    synthesis_time: float = 0.0
    critique_time: float = 0.0
    research_time: float = 0.0

class ResearchOutput(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    summary: str
    sections: List[Section]
    sources: List[Source]
    critique: Critique
    metadata: Metadata

# Internal Messaging Schemas

class Plan(BaseModel):
    sub_queries: List[str]
    search_strategy: str

class SearchResult(BaseModel):
    sub_query: str
    sources: List[Source]

class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    sender: str
    recipient: str
    msg_type: str
    payload: dict
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
