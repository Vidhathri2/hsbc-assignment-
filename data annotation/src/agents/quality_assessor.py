from typing import List
from src.schema import Sample, AnnotationResult
from src.llm import generate_structured_response
from src.logger import get_logger

logger = get_logger("QualityAssessorAgent")

class QualityAssessorAgent:
    def __init__(self, client, confidence_threshold: float = 0.8):
        self.client = client
        self.confidence_threshold = confidence_threshold

    def assess(self, samples: List[Sample]) -> List[Sample]:
        logger.info(f"Assessing {len(samples)} samples...")
        for sample in samples:
            if sample.confidence is not None and sample.confidence < self.confidence_threshold:
                logger.info(f"Sample {sample.id} has low confidence ({sample.confidence}). Re-assessing...")
                self._re_evaluate(sample)
            else:
                sample.is_assessed = True
        return samples

    def _re_evaluate(self, sample: Sample):
        prompt = (
            "You are a Senior Quality Assurance Annotator. The previous annotator was unsure about the classification "
            "of the following news article. Read it carefully, think step-by-step about its primary subject matter, "
            "and provide the most accurate category label (e.g., Politics, Sports, Technology, Business, Entertainment, Health, Science).\n"
            "Provide the predicted label and your confidence score (0.0 to 1.0).\n\n"
            f"Article:\n{sample.text}\n\n"
            f"Previous Label: {sample.label} (Confidence: {sample.confidence})\n"
        )
        
        try:
            result = generate_structured_response(self.client, prompt, AnnotationResult)
            sample.label = result.label
            sample.confidence = result.confidence
            sample.is_assessed = True
            logger.info(f"Re-assessed sample {sample.id}: New Label={sample.label}, New Confidence={sample.confidence}")
        except Exception as e:
            logger.error(f"Error during QA re-evaluation for sample {sample.id}: {e}")

    def all_above_threshold(self, samples: List[Sample]) -> bool:
        return all(s.confidence is not None and s.confidence >= self.confidence_threshold for s in samples)
