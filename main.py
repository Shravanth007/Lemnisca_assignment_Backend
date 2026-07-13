"""
ClearPath Chatbot – FastAPI Application
=========================================
Exposes:
  POST /query   →  main chat endpoint (matches API_CONTRACT.md)
  GET  /health  →  liveness check

Start with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.retrieval import retrieve, reload_index
from rag.database import index_exists
from rag.ingest import build_index
from router.classifier import route_query, log_request
from evaluator.checks import evaluate
from models.groq_client import generate_answer


app = FastAPI(
    title="ClearPath Chatbot API",
    description="RAG-powered customer support chatbot for ClearPath PM SaaS",
    version="1.0.0",
)

_default_origins = [
    "https://clear-path-frontend.vercel.app",
    "http://localhost:3000",
]
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
allowed_origins = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    if not index_exists():
        print("🔧 BM25 index not found — attempting to build …")
        try:
            build_index()
        except Exception as e:
            print(f"⚠️ Could not build index (read-only FS?): {e}")
            print("   Make sure the pre-built index is committed to the repo.")
    else:
        print("⚡ BM25 index loaded from disk.")



class QueryRequest(BaseModel):
    question: str = Field(..., max_length=500)
    conversation_id: Optional[str] = None


class TokenInfo(BaseModel):
    input: int
    output: int


class MetadataResponse(BaseModel):
    model_used: str
    classification: str
    tokens: TokenInfo
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: List[str]


class SourceResponse(BaseModel):
    document: str
    page: int
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    metadata: MetadataResponse
    sources: List[SourceResponse]
    conversation_id: str



_conversations: Dict[str, List[Dict[str, str]]] = {}


# ponytail: in-memory per-IP limiter, single-instance only; move to slowapi/redis if scaled out
_RATE_LIMIT = 10
_RATE_WINDOW_SECONDS = 60
_request_log: Dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # X-Forwarded-For is spoofable, but here it only lets someone shard their own rate limit.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(ip: str) -> None:
    now = time.time()
    timestamps = _request_log[ip]
    while timestamps and now - timestamps[0] > _RATE_WINDOW_SECONDS:
        timestamps.popleft()
    if len(timestamps) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    timestamps.append(now)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, request: Request):

    _enforce_rate_limit(_client_ip(request))

    start = time.perf_counter()

    conv_id = req.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    routing = route_query(req.question)
    classification = routing["classification"]
    model = routing["model"]

    chunks = retrieve(req.question)
    chunks_retrieved = len(chunks)

    history = _conversations.get(conv_id, [])
    llm_result = generate_answer(
        question=req.question,
        chunks=chunks,
        model=model,
        conversation_history=history[-6:], 
    )
    answer = llm_result["answer"]
    tokens_in = llm_result["tokens_input"]
    tokens_out = llm_result["tokens_output"]

    _conversations.setdefault(conv_id, []).append(
        {"role": "user", "content": req.question}
    )
    _conversations[conv_id].append(
        {"role": "assistant", "content": answer}
    )

    sources = [
        {
            "document": c["source"],
            "page": c["page"],
            "relevance_score": c["relevance_score"],
        }
        for c in chunks
    ]

    flags = evaluate(
        answer=answer,
        chunks_retrieved=chunks_retrieved,
        sources=sources,
        classification=classification,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    log_request(
        query=req.question,
        classification=classification,
        model_used=model,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=answer,
        metadata=MetadataResponse(
            model_used=model,
            classification=classification,
            tokens=TokenInfo(input=tokens_in, output=tokens_out),
            latency_ms=latency_ms,
            chunks_retrieved=chunks_retrieved,
            evaluator_flags=flags,
        ),
        sources=[SourceResponse(**s) for s in sources],
        conversation_id=conv_id,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": index_exists()}
