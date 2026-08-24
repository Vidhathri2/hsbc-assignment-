# Enterprise Multi-Agent Autonomous Data Annotation Pipeline

## 🚀 Architectural Overview
This pipeline represents a state-of-the-art **Agentic AI Architecture** built entirely on the official **LangChain** and **LangGraph** frameworks. It operates as a fully autonomous data engine designed to ingest raw data, evaluate it, and train machine learning models iteratively.

### 🧠 Core Frameworks
* **Orchestration**: `langgraph` (Hierarchical `StateGraph` routing)
* **Agent Logic**: `langchain`, `langchain_core` (PromptTemplates & Structured Output Chains)
* **LLM Engine**: `langchain_google_genai` (`gemini-3.6-flash`)
* **Vector Store / RAG**: `langchain_community` (`TFIDFRetriever`)

---

## 🤖 The Agentic Hierarchy (LangGraph)
Unlike standard iterative scripts, this pipeline uses a **Supervisor Agent Pattern** built on LangGraph. 
A central `Supervisor Node` manages the global `PipelineState` and dynamically delegates execution to specialized worker agents based on real-time conditions.

### The Specialized Agents:
1. **Selection Agent (`select_data`)**: Uses **Shannon Entropy** to evaluate the ML model's confusion matrix, actively pulling the highest-value data from the raw pool (Active Learning).
2. **Annotator Agent (`annotate`)**: Uses `gemini-3.6-flash` to autonomously read, parse, and categorize the raw text, outputting strictly validated Pydantic JSON via LangChain chains.
3. **Quality Assessor (`qa`)**: Acts as a self-reflection node. It evaluates the Annotator's confidence scores. If confidence falls below 80%, the QA agent forces a re-evaluation chain to ensure dataset purity.
4. **Trainer Agent (`train`)**: Evaluates classic ML algorithms (`RandomForest`, `LogisticRegression`, `KNN`) against the newly labelled pool, scoring them on Accuracy and F1 metrics.

---

## 📚 Agentic Knowledge Base (RAG)
Once the Active Learning loop hits its target metrics, the labelled dataset is instantly compiled into native LangChain `Document` chunks.
It utilizes an ultra-fast, in-memory `TFIDFRetriever` to facilitate real-time chat, allowing the user to query the newly synthesized knowledge base natively.

---

## ⚙️ How to Run

1. **Environment Setup**: Ensure you have a `.env` file in the root directory containing your API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute Pipeline**:
   ```bash
   python main.py
   ```
Watch the terminal as the LangGraph Supervisor coordinates the Gemini agents in real-time!
