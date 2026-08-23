from pydantic import BaseModel, Field
from typing import List, Optional

class Sample(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    summary: Optional[str] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    is_assessed: bool = False

class AnnotationResult(BaseModel):
    label: str = Field(description="The predicted category for the news article (e.g., Politics, Sports, Technology, Business, Entertainment).")
    confidence: float = Field(description="Confidence score of the prediction between 0.0 and 1.0.")

class BatchAnnotationResult(BaseModel):
    results: List[AnnotationResult] = Field(description="List of annotation results corresponding to the input batch.")
