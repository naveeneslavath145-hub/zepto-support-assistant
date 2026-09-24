
import os
from typing import TypedDict, List

import chromadb

from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from fastapi import FastAPI

os.environ["MOCK_LLM"] = os.getenv(
    "MOCK_LLM",
    "1"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="support_assistant/chroma_db"
)

collection = chroma_client.get_collection(
    name="policy_collection"
)

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "track",
    "cancel",
    "gift card",
    "support hours"
]

def retrieve_documents(
    query,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    retrieved = []

    for i in range(
        len(results["ids"][0])
    ):

        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "document_id": results["metadatas"][0][i][
                "document_id"
            ]
        })

    return retrieved

def classify_query(query: str):

    query_lower = query.lower()

    for keyword in POLICY_KEYWORDS:

        if keyword in query_lower:
            return "policy_question"

    return "general_question"

class SupportState(TypedDict):

    query: str
    intent: str
    retrieved_chunks: list
    answer: str
    sources: list
    confidence: float

def classify_intent(state):

    return {
        "intent": classify_query(
            state["query"]
        )
    }

def retrieve_and_answer(state):

    retrieved = retrieve_documents(
        state["query"],
        top_k=3
    )

    if retrieved:

        answer = (
            "Based on the retrieved context: "
            + retrieved[0]["text"]
        )

        confidence = 0.90

    else:

        answer = (
            "I could not find relevant policy information."
        )

        confidence = 0.20


    return {
        "retrieved_chunks": retrieved,
        "answer": answer,
        "sources": [
            item["chunk_id"]
            for item in retrieved
        ],
        "confidence": confidence
    }

def direct_answer(state):

    return {
        "answer":
            "I can only answer questions about "
            "Zepto policies right now.",

        "sources": [],

        "confidence": 1.0
    }

def route_by_intent(state):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"

graph_builder = StateGraph(
    SupportState
)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer"
    }

)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

support_graph = graph_builder.compile()

class SupportResponse(BaseModel):

    answer: str

    sources: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

def ask_assistant(query):

    state = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }

    result = support_graph.invoke(
        state
    )

    return SupportResponse(
        answer=result["answer"],
        sources=result.get(
            "sources",
            []
        ),
        confidence=result.get(
            "confidence",
            0.0
        )
    )

app = FastAPI(
    title="Zepto Policy Support Assistant",
    version="1.0.0"
)

class AskRequest(BaseModel):

    query: str

@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask_endpoint(request: AskRequest):

    return ask_assistant(
        request.query
    )

