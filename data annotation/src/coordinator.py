from typing import List, TypedDict, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from src.schema import Sample
from src.agents.annotator import AnnotatorAgent
from src.agents.quality_assessor import QualityAssessorAgent
from src.agents.trainer import TrainerAgent
from src.logger import get_logger

logger = get_logger("Coordinator")

# Define the state for the Hierarchical LangGraph
class PipelineState(TypedDict):
    unlabelled_pool: List[Sample]
    labelled_pool: List[Sample]
    current_batch: List[Sample]
    iteration: int
    target_reached: bool
    batch_size: int
    max_iterations: int
    next_agent: str
    qa_attempts: int

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
        
        # ---------------------------------------------------------
        # THE SUPERVISOR NODE (Hierarchy Leader)
        # ---------------------------------------------------------
        def supervisor_node(state: PipelineState) -> Dict:
            """The hierarchical supervisor that delegates tasks to sub-agents."""
            # 1. Check exit conditions
            if state["target_reached"]:
                logger.info("Supervisor: Target accuracy reached. Terminating pipeline.")
                return {"next_agent": "END"}
            if state["iteration"] >= state["max_iterations"] or not state["unlabelled_pool"]:
                logger.info("Supervisor: Max iterations reached or pool exhausted.")
                return {"next_agent": "END"}
                
            # 2. Delegate to Selection Agent (if we have no active batch)
            if not state["current_batch"]:
                return {"next_agent": "select_data", "iteration": state["iteration"] + 1, "qa_attempts": 0}
                
            # 3. Delegate to Annotator Agent (if batch is unlabelled)
            if any(s.label is None for s in state["current_batch"]):
                return {"next_agent": "annotate"}
                
            # 4. Delegate to QA Agent (if batch hasn't passed QA)
            # QA assesses confidence. If attempts < 3, keep trying.
            if state["qa_attempts"] < 3 and any(not s.is_assessed for s in state["current_batch"]):
                return {"next_agent": "qa"}
                
            # 5. Delegate to Trainer Agent (all QA passed or max QA attempts reached)
            return {"next_agent": "train"}

        # ---------------------------------------------------------
        # THE WORKER NODES (Sub-Agents)
        # ---------------------------------------------------------
        def select_data_node(state: PipelineState) -> Dict:
            logger.info(f"--- Hierarchical Iteration {state['iteration']} ---")
            logger.info("Supervisor -> Selection Agent: Fetching next optimal batch.")
            candidate_pool = state['unlabelled_pool']
            
            # Active Learning Entropy
            if self.trainer.best_model:
                entropies = self.trainer.get_uncertainties(state['unlabelled_pool'])
                paired = sorted(zip(entropies, state['unlabelled_pool']), key=lambda x: x[0], reverse=True)
                candidate_pool = [s for e, s in paired][:state['batch_size'] * 3]

            selected_samples = self.annotator.select_samples(
                unlabelled_pool=candidate_pool, 
                labelled_pool=state['labelled_pool'], 
                batch_size=state['batch_size']
            )
            
            # Remove from unlabelled pool
            unlabelled = [s for s in state['unlabelled_pool'] if s not in selected_samples]
            return {"unlabelled_pool": unlabelled, "current_batch": selected_samples}

        def annotate_node(state: PipelineState) -> Dict:
            logger.info("Supervisor -> Annotator Agent: Labelling batch.")
            annotated_batch = self.annotator.annotate(state['current_batch'])
            return {"current_batch": annotated_batch}

        def qa_node(state: PipelineState) -> Dict:
            logger.info(f"Supervisor -> QA Agent: Reviewing batch (Attempt {state['qa_attempts'] + 1}).")
            assessed_batch = self.qa_agent.assess(state['current_batch'])
            
            if self.qa_agent.all_above_threshold(assessed_batch):
                logger.info("QA Agent: All samples passed confidence threshold.")
                # Force all to be marked assessed so we can move to train
                for s in assessed_batch: s.is_assessed = True
            
            return {"current_batch": assessed_batch, "qa_attempts": state["qa_attempts"] + 1}

        def train_node(state: PipelineState) -> Dict:
            logger.info("Supervisor -> Trainer Agent: Retraining Model.")
            new_labelled = state['labelled_pool'] + state['current_batch']
            logger.info(f"Labelled pool now contains {len(new_labelled)} samples.")
            
            target_reached, metrics = self.trainer.train_and_evaluate(new_labelled)
            # Clear batch so supervisor starts over
            return {
                "labelled_pool": new_labelled, 
                "target_reached": target_reached,
                "current_batch": []
            }

        # Build Graph
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("select_data", select_data_node)
        workflow.add_node("annotate", annotate_node)
        workflow.add_node("qa", qa_node)
        workflow.add_node("train", train_node)

        # Star Architecture: Everything routes through Supervisor
        workflow.set_entry_point("supervisor")
        
        # Define hierarchical conditional router
        def router(state: PipelineState):
            if state["next_agent"] == "END": return END
            return state["next_agent"]

        workflow.add_conditional_edges("supervisor", router)
        
        # All workers report back to the supervisor
        workflow.add_edge("select_data", "supervisor")
        workflow.add_edge("annotate", "supervisor")
        workflow.add_edge("qa", "supervisor")
        workflow.add_edge("train", "supervisor")
        
        return workflow.compile()

    def run_pipeline(self, batch_size: int = 10, max_iterations: int = 20):
        initial_state = PipelineState(
            unlabelled_pool=self.unlabelled_pool,
            labelled_pool=self.labelled_pool,
            current_batch=[],
            iteration=0,
            target_reached=False,
            batch_size=batch_size,
            max_iterations=max_iterations,
            next_agent="select_data",
            qa_attempts=0
        )
        
        logger.info("Initializing Agentic Hierarchy: Supervisor online.")
        final_state = self.workflow.invoke(initial_state)
        
        # Sync state
        self.unlabelled_pool = final_state["unlabelled_pool"]
        self.labelled_pool = final_state["labelled_pool"]
        
        return self.trainer.best_model, self.labelled_pool
