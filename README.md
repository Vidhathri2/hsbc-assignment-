# HSBC Agentic AI Assignment

This repository contains the complete deliverables for the Agentic AI Pipeline assignment. 

## Project Structure

The workspace is divided into three core multi-agent systems, each designed to solve complex autonomous tasks using state-of-the-art LLM architectures (Gemini Flash) and Agentic frameworks.

1. **`/data annotation`** (Core Assignment)
   - A fully autonomous **Multi-Agent Active Learning Pipeline**.
   - Includes Annotator, QA, Trainer, and Indexer/Chat Agents.
   - Features real-time RSS data ingestion, Entropy-based uncertainty sampling, and a local TF-IDF semantic RAG search engine.

2. **`/web-intel-convo`**
   - Web Intelligence Conversational RAG system.
   
3. **`/web-research-agent`**
   - A distributed Multi-Agent Research System using robust message-bus coordination for enterprise-grade autonomous data synthesis.

## Setup & Execution

Each folder contains its own self-contained environment logic. To run the primary data annotation pipeline:

```powershell
cd "data annotation"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
