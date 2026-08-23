from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.schema import Sample, BatchAnnotationResult
from src.llm import generate_structured_response
from src.logger import get_logger

logger = get_logger("AnnotatorAgent")

class AnnotatorAgent:
    def __init__(self, client, token_budget: int = 50000):
        self.client = client
        self.token_budget = token_budget
        self.tokens_used = 0
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def select_samples(self, unlabelled_pool: List[Sample], labelled_pool: List[Sample], batch_size: int = 5) -> List[Sample]:
        if not labelled_pool or not unlabelled_pool:
            return unlabelled_pool[:batch_size]

        all_texts = [s.text for s in labelled_pool] + [s.text for s in unlabelled_pool]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        labelled_tfidf = tfidf_matrix[:len(labelled_pool)]
        unlabelled_tfidf = tfidf_matrix[len(labelled_pool):]

        sim_matrix = cosine_similarity(unlabelled_tfidf, labelled_tfidf)
        max_sim_to_labelled = np.max(sim_matrix, axis=1)

        least_similar_indices = np.argsort(max_sim_to_labelled)[:batch_size]
        
        selected_samples = [unlabelled_pool[i] for i in least_similar_indices]
        return selected_samples

    def annotate(self, samples: List[Sample]) -> List[Sample]:
        if self.tokens_used >= self.token_budget:
            logger.warning("Token budget exhausted. Cannot annotate more samples.")
            return samples

        prompt = (
            "You are an expert data annotator. Please classify the following news articles into appropriate categories "
            "(e.g., Politics, Sports, Technology, Business, Entertainment, Health, Science).\n"
            "Provide the predicted label and a confidence score between 0.0 and 1.0 for each article.\n\n"
        )
        
        for i, sample in enumerate(samples):
            prompt += f"Article {i+1}:\n{sample.text}\n\n"

        logger.info(f"Annotating {len(samples)} samples...")
        try:
            result = generate_structured_response(self.client, prompt, BatchAnnotationResult)
            
            for i, ann in enumerate(result.results):
                if i < len(samples):
                    samples[i].label = ann.label
                    samples[i].confidence = ann.confidence
                    samples[i].is_assessed = False
            
            self.tokens_used += len(prompt) // 4 + 100
            
        except Exception as e:
            logger.error(f"Error during annotation: {e}")

        return samples
