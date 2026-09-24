# Module 3 — Support Assistant

## Architecture

Policy documents
→ text chunking
→ Sentence Transformer embeddings
→ ChromaDB
→ query embedding
→ top-3 cosine-similarity retrieval
→ LangGraph routing
→ Pydantic validation
→ FastAPI `/ask`

## Embedding model

`all-MiniLM-L6-v2`

The embedding model runs locally.

## LangGraph nodes

The graph contains three required named nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

## Intent classification

A query is classified as `policy_question` when its lowercase
text contains one of:

* delivery
* return
* refund
* membership
* tracking
* cancel
* gift card
* support hours

All other queries are classified as `general_question`.

## Mock LLM

`MOCK_LLM=1` is the default configuration.

In mock mode, policy answers are deterministic:

`Based on the retrieved context: {top_chunk_snippet}`

General questions return:

`I can only answer questions about Zepto policies right now.`

No external LLM API is required.

## FastAPI

Start the API with:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoint:

`POST /ask`

Example request:

```json
{
  "query": "How can I track my order?"
}
```

## Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run -p 8000:8000 zepto-support-assistant
```