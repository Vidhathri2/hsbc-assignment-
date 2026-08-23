from typing import List
from src.schema import Sample
from src.agents.annotator import AnnotatorAgent
from src.agents.quality_assessor import QualityAssessorAgent
from src.agents.trainer import TrainerAgent
from src.logger import get_logger

logger = get_logger("Coordinator")

class Coordinator:
    def __init__(self, 
                 annotator: AnnotatorAgent, 
                 qa_agent: QualityAssessorAgent, 
                 trainer: TrainerAgent):
        self.annotator = annotator
        self.qa_agent = qa_agent
        self.trainer = trainer
        
        self.unlabelled_pool: List[Sample] = []
        self.labelled_pool: List[Sample] = []

    def load_data(self, samples: List[Sample]):
        self.unlabelled_pool.extend(samples)
        logger.info(f"Loaded {len(samples)} samples into the unlabelled pool.")

    def run_pipeline(self, batch_size: int = 10, max_iterations: int = 20):
        iteration = 0
        target_reached = False

        while iteration < max_iterations and not target_reached and self.unlabelled_pool:
            iteration += 1
            logger.info(f"--- Pipeline Iteration {iteration} ---")
            
            # 1. Active Learning: Calculate uncertainty if a model exists
            candidate_pool = self.unlabelled_pool
            if self.trainer.best_model:
                logger.info("Calculating prediction entropy to find most uncertain samples...")
                entropies = self.trainer.get_uncertainties(self.unlabelled_pool)
                # Sort by highest entropy
                paired = sorted(zip(entropies, self.unlabelled_pool), key=lambda x: x[0], reverse=True)
                candidate_pool = [s for e, s in paired][:batch_size * 3] # take top candidates

            # 2. Annotator selects and labels a batch from the uncertain candidates
            selected_samples = self.annotator.select_samples(
                unlabelled_pool=candidate_pool, 
                labelled_pool=self.labelled_pool, 
                batch_size=batch_size
            )
            
            # Remove selected from unlabelled pool immediately
            for s in selected_samples:
                if s in self.unlabelled_pool:
                    self.unlabelled_pool.remove(s)

            annotated_batch = self.annotator.annotate(selected_samples)

            # 2. Quality Assurance loop until batch confidence is high
            qa_attempts = 0
            while qa_attempts < 3: # prevent infinite loop
                assessed_batch = self.qa_agent.assess(annotated_batch)
                if self.qa_agent.all_above_threshold(assessed_batch):
                    logger.info("All samples in batch passed QA confidence threshold.")
                    break
                logger.warning("Some samples still below threshold after QA, retrying QA...")
                annotated_batch = assessed_batch
                qa_attempts += 1
            
            # 3. Add to labelled pool
            self.labelled_pool.extend(annotated_batch)
            logger.info(f"Labelled pool now contains {len(self.labelled_pool)} samples.")

            # 4. Train and Evaluate
            target_reached, metrics = self.trainer.train_and_evaluate(self.labelled_pool)
            
            if target_reached:
                logger.info("Active Learning Pipeline Completed Successfully!")
                break

        if not target_reached:
            logger.info("Max iterations reached or unlabelled pool exhausted before reaching target accuracy.")
            
        return self.trainer.best_model, self.labelled_pool
