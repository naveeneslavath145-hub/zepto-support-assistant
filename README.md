\# Zepto Data \& AI Platform — AI/ML Capstone



This repository contains the three modules developed for the Zepto Data \& AI Platform capstone project.



\## Project Structure



\- `data\_pipeline/` — Book data scraping, cleaning, SQLite storage, and SQL analysis.

\- `analytics/` — Titanic EDA, preprocessing, classification, imbalance analysis, hyperparameter tuning, and regression.

\- `support\_assistant/` — Zepto policy RAG support assistant using embeddings, ChromaDB, LangGraph, and FastAPI.



\## Setup



Install the required packages for each module using its `requirements.txt` where provided.



\## Running the Modules



\### 1. Data Pipeline



Open and run:



`data\_pipeline/data\_pipeline.ipynb`



The notebook performs scraping, cleaning, SQLite storage, SQL queries, and pandas analysis.



\### 2. Analytics



Open and run:



`analytics/analytics.ipynb`



The notebook performs Titanic EDA, preprocessing, classification, model comparison, imbalance analysis, Random Forest tuning, and regression.



\### 3. Support Assistant



From the project root:



```bash

cd support\_assistant

pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000


## Submission
This repository contains the complete three-module AI/ML capstone implementation.
