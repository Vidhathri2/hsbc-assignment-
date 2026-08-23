# Enterprise Multi-Agent Autonomous Data Annotation Pipeline

## 🚀 Architectural Overview
This pipeline represents a state-of-the-art **Agentic AI Architecture** built entirely on the official **LangChain** and **LangGraph** frameworks. It operates as a local, fully autonomous data engine designed to ingest raw data, evaluate it, and train machine learning models iteratively.

To ensure strict privacy, zero cost, and immunity to API rate limits (429 RESOURCE_EXHAUSTED), this system operates **100% locally** using **Llama 2 via Ollama**.

### 🧠 Core Frameworks
* **Orchestration**: `langgraph` (Hierarchical `StateGraph` routing)
* **Agent Logic**: `langchain`, `langchain_core` (PromptTemplates & Structured Output Chains)
* **LLM Engine**: `langchain_ollama` (`llama2:latest`)
* **Vector Store / RAG**: `langchain_community` (`TFIDFRetriever`)

---

## 🤖 The Agentic Hierarchy (LangGraph)
Unlike standard iterative scripts, this pipeline uses a **Supervisor Agent Pattern** built on LangGraph. 
A central `Supervisor Node` manages the global `PipelineState` and dynamically delegates execution to specialized worker agents based on real-time conditions.

### The Specialized Agents:
1. **Selection Agent (`select_data`)**: Uses **Shannon Entropy** to evaluate the ML model's confusion matrix, actively pulling the highest-value data from the raw pool (Active Learning).
2. **Annotator Agent (`annotate`)**: Uses `llama2` to autonomously read, parse, and categorize the raw text, outputting strictly validated Pydantic JSON via LangChain chains.
3. **Quality Assessor (`qa`)**: Acts as a self-reflection node. It evaluates the Annotator's confidence scores. If confidence falls below 80%, the QA agent forces a re-evaluation chain to ensure dataset purity.
4. **Trainer Agent (`train`)**: Evaluates classic ML algorithms (`RandomForest`, `LogisticRegression`, `KNN`) against the newly labelled pool, scoring them on Accuracy and F1 metrics.

---

## 📚 Agentic Knowledge Base (RAG)
Once the Active Learning loop hits its target metrics, the labelled dataset is instantly compiled into native LangChain `Document` chunks.
It utilizes an ultra-fast, in-memory `TFIDFRetriever` to facilitate real-time chat, allowing the user to query the newly synthesized knowledge base natively.

---

## ⚙️ How to Run
This repository is pre-configured to run entirely on local hardware.

1. **Prerequisites**: Ensure you have [Ollama](https://ollama.com/) installed and running locally with the `llama2` model (`ollama run llama2`).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute Pipeline**:
   ```bash
   python main.py
   ```
Watch the terminal as the LangGraph Supervisor coordinates the Llama 2 agents in real-time!
