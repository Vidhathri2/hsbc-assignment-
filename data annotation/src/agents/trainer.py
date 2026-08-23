from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np

from src.schema import Sample
from src.logger import get_logger

logger = get_logger("TrainerAgent")

class TrainerAgent:
    def __init__(self, target_accuracy: float = 0.85):
        self.target_accuracy = target_accuracy
        self.best_model = None
        self.best_score = 0.0

    def train_and_evaluate(self, samples: List[Sample]) -> Tuple[bool, Dict[str, Any]]:
        if len(samples) < 10:
            logger.warning("Not enough samples to train a robust model (need at least 10). Waiting for more data...")
            return False, {}

        # Prepare data
        texts = [s.text for s in samples]
        labels = [s.label for s in samples]

        # 1. Divide into train(60%)/eval(20%)/test(20%)
        # Note: If classes are too few, stratified split might fail. We use a simple split here.
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            X_train, X_eval, y_train, y_eval = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
        except ValueError as e:
            logger.error(f"Not enough class diversity for splitting: {e}")
            return False, {}

        logger.info(f"Training on {len(X_train)} samples, evaluating on {len(X_eval)}, testing on {len(X_test)}.")

        from sklearn.neighbors import KNeighborsClassifier
        # 2. Define models to try (including KNN as suggested by assignment)
        models_to_try = {
            "KNN": Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', KNeighborsClassifier(n_neighbors=3))
            ]),
            "LogisticRegression": Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LogisticRegression(max_iter=1000))
            ]),
            "RandomForest": Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        }

        best_eval_f1 = -1.0
        best_model_name = None
        best_model_instance = None

        # 3. Train and evaluate
        for name, model in models_to_try.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_eval)
                # Use macro average as there could be multiple classes
                f1 = f1_score(y_eval, preds, average='macro', zero_division=0)
                
                logger.info(f"Model {name} Eval F1: {f1:.4f}")
                
                if f1 > best_eval_f1:
                    best_eval_f1 = f1
                    best_model_name = name
                    best_model_instance = model
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        if not best_model_instance:
            return False, {}

        # 4. Test the best model
        logger.info(f"Testing Best Model ({best_model_name})...")
        test_preds = best_model_instance.predict(X_test)
        
        test_precision = precision_score(y_test, test_preds, average='macro', zero_division=0)
        test_recall = recall_score(y_test, test_preds, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
        test_accuracy = accuracy_score(y_test, test_preds)

        metrics = {
            "model_name": best_model_name,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }

        logger.info(f"Test Metrics - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        self.best_model = best_model_instance
        self.best_score = test_accuracy

        # 5. Check early stopping condition
        target_reached = test_accuracy >= self.target_accuracy
        if target_reached:
            logger.info(f"Target accuracy ({self.target_accuracy}) reached! Stopping training process.")
        
        return target_reached, metrics

    def get_uncertainties(self, unlabelled_samples: List[Sample]) -> List[float]:
        """Calculates prediction entropy to find most uncertain samples (Active Learning)."""
        if not self.best_model or not unlabelled_samples:
            return [1.0] * len(unlabelled_samples)
            
        texts = [s.text for s in unlabelled_samples]
        try:
            probs = self.best_model.predict_proba(texts)
            # Calculate entropy: -sum(p * log(p))
            # The higher the entropy, the more uncertain the model is
            entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropies.tolist()
        except Exception as e:
            logger.error(f"Could not calculate uncertainty: {e}")
            return [1.0] * len(unlabelled_samples)

