from typing import List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from src.schema import Sample
from src.agents.annotator import AnnotatorAgent
from src.agents.quality_assessor import QualityAssessorAgent
from src.agents.trainer import TrainerAgent
from src.logger import get_logger

logger = get_logger("Coordinator")

# Define the state for LangGraph
class PipelineState(TypedDict):
    unlabelled_pool: List[Sample]
    labelled_pool: List[Sample]
    current_batch: List[Sample]
    iteration: int
    target_reached: bool
    batch_size: int
    max_iterations: int

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
        
        self.workflow = self._build_graph()

    def load_data(self, samples: List[Sample]):
        self.unlabelled_pool.extend(samples)
        logger.info(f"Loaded {len(samples)} samples into the unlabelled pool.")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PipelineState)
        
        # Define Nodes
        def select_data_node(state: PipelineState) -> Dict:
            logger.info(f"--- LangGraph Iteration {state['iteration'] + 1} ---")
            candidate_pool = state['unlabelled_pool']
            
            # Active Learning
            if self.trainer.best_model:
                logger.info("Calculating prediction entropy to find most uncertain samples...")
                entropies = self.trainer.get_uncertainties(state['unlabelled_pool'])
                paired = sorted(zip(entropies, state['unlabelled_pool']), key=lambda x: x[0], reverse=True)
                candidate_pool = [s for e, s in paired][:state['batch_size'] * 3]

            selected_samples = self.annotator.select_samples(
                unlabelled_pool=candidate_pool, 
                labelled_pool=state['labelled_pool'], 
                batch_size=state['batch_size']
            )
            
            # Remove from unlabelled
            unlabelled = [s for s in state['unlabelled_pool'] if s not in selected_samples]
            return {"unlabelled_pool": unlabelled, "current_batch": selected_samples}

        def annotate_node(state: PipelineState) -> Dict:
            annotated_batch = self.annotator.annotate(state['current_batch'])
            return {"current_batch": annotated_batch}

        def qa_node(state: PipelineState) -> Dict:
            assessed_batch = self.qa_agent.assess(state['current_batch'])
            # We assume it assesses until it passes internally (like our QA agent does)
            return {"current_batch": assessed_batch}

        def train_node(state: PipelineState) -> Dict:
            new_labelled = state['labelled_pool'] + state['current_batch']
            logger.info(f"Labelled pool now contains {len(new_labelled)} samples.")
            
            target_reached, metrics = self.trainer.train_and_evaluate(new_labelled)
            return {
                "labelled_pool": new_labelled, 
                "target_reached": target_reached,
                "iteration": state['iteration'] + 1
            }

        # Add Nodes
        workflow.add_node("select_data", select_data_node)
        workflow.add_node("annotate", annotate_node)
        workflow.add_node("qa", qa_node)
        workflow.add_node("train", train_node)

        # Define Edges
        workflow.set_entry_point("select_data")
        workflow.add_edge("select_data", "annotate")
        workflow.add_edge("annotate", "qa")
        workflow.add_edge("qa", "train")

        # Define Conditional Logic
        def should_continue(state: PipelineState) -> str:
            if state["target_reached"]:
                logger.info("LangGraph: Active Learning Pipeline Completed Successfully!")
                return END
            if state["iteration"] >= state["max_iterations"] or not state["unlabelled_pool"]:
                logger.info("LangGraph: Max iterations reached or pool exhausted.")
                return END
            return "select_data"

        workflow.add_conditional_edges("train", should_continue)
        
        return workflow.compile()

    def run_pipeline(self, batch_size: int = 10, max_iterations: int = 20):
        initial_state = PipelineState(
            unlabelled_pool=self.unlabelled_pool,
            labelled_pool=self.labelled_pool,
            current_batch=[],
            iteration=0,
            target_reached=False,
            batch_size=batch_size,
            max_iterations=max_iterations
        )
        
        # Execute the LangGraph workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Sync state back to class for external access
        self.unlabelled_pool = final_state["unlabelled_pool"]
        self.labelled_pool = final_state["labelled_pool"]
        
        return self.trainer.best_model, self.labelled_pool
