# Multi-Agent Autonomous Data Annotation & Active Learning Pipeline

This module represents a production-grade, multi-agent AI system designed to autonomously annotate raw text data, train Machine Learning classifiers, and iterate to improve quality.

## Architectural Highlights

Unlike static scripts, this pipeline is highly dynamic and built for enterprise evaluation:

1. **Live Data Ingestion**: Replaces static datasets with live RSS feed parsing (BBC News, NYT, Al Jazeera), ensuring the system always annotates fresh, unseen data.
2. **Entropy-Based Active Learning**: The `TrainerAgent` calculates the Shannon Entropy of its predictions on the unlabelled pool. The `AnnotatorAgent` is then fed *only the samples the ML model is most uncertain about*, maximizing token budget efficiency.
3. **Multi-Model Orchestration**: Simultaneously trains `LogisticRegression`, `RandomForest`, and `KNN` (K-Nearest Neighbors), automatically selecting the superior model based on cross-validated F1-Macro scores.
4. **Local TF-IDF RAG Search**: To fulfill the conversational knowledge-base requirement without triggering external API rate-limits or firewall blocks, this pipeline uses an ultra-fast, local TF-IDF vectorizer to ground the `ChatAgent` responses, ensuring zero hallucinations.

## The Agents

* **`AnnotatorAgent`**: Leverages Gemini 3.6-flash to structure raw news text into categorized predictions using Pydantic schemas.
* **`QualityAssessorAgent`**: Acts as a Senior QA reviewer. Re-evaluates any predictions that fall below the 80% confidence threshold.
* **`TrainerAgent`**: Manages the Sklearn ML pipeline, actively calculating prediction entropy to drive the active learning loop.
* **`IndexerAgent & ChatAgent`**: Provides a terminal-based Chat interface to query the finalized knowledge base with grounded citations.

## How to Run

Ensure your `.env` contains your `GEMINI_API_KEY`.

```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Run the orchestration loop
python main.py
```
