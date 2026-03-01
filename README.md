# ClearPath Chatbot - Backend API

A RAG-powered customer support chatbot built with FastAPI for the ClearPath PM SaaS platform. The backend has three layers:

- **BM25 Retrieval Pipeline** -- Extracts and chunks 30 PDF documents, builds a BM25Okapi index, and retrieves the top 5 relevant chunks per query.
- **Deterministic Router** -- A rule-based classifier that routes simple queries to a smaller model and complex queries to a larger model based on keyword matching, word count, and sentence structure.
- **Output Evaluator** -- Inspects every LLM response and flags issues like missing context, refusals, or internal data leaks.

---

## Local Setup

### 1. Clone the repository

```bash
git clone $repo_link
cd backend
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the `backend` directory:

```
Groq_Llama_33_70b_versatile=your_api_key_here
Groq_Llama_31_8b_instant=your_api_key_here
```

### 5. Build the BM25 index

```bash
python -m rag.ingest
```

This reads all PDFs from `rag/ClearPath/clearpath_docs/`, chunks them, and saves the BM25 index to `rag/index_store/`.

### 6. Start the server

```bash
uvicorn main:app --reload --port 8000
```

---

## Models Used

Two Groq-hosted LLMs are used, selected at runtime by the deterministic router:

- `llama-3.1-8b-instant` -- For simple queries (short, direct questions).
- `llama-3.3-70b-versatile` -- For complex queries (comparisons, multi-part questions, analytical requests).

---

## Bonus Challenges Attempted

### Conversation Memory

Conversation history is maintained in-memory using a `conversation_id`. Each new session gets a unique ID, and follow-up messages with the same ID continue the conversation. The backend passes the last 3 exchanges (6 messages) to the LLM for context.

### Live Deploy

The frontend is deployed on Vercel and is live at:
https://clear-path-frontend.vercel.app/

The backend is configured for Vercel deployment using `vercel.json`.

---

## Known Issues and Limitations

- **Aggressive Routing** -- The deterministic router is strict to prioritize answer quality. Keywords like "how do", "why", or "explain" immediately trigger the complex classification, so even basic questions like "how do I reset my password" get routed to the 70B model.

- **Ephemeral Memory** -- The conversation store is a Python dictionary (`_conversations`) held in memory. All conversation history is lost when the server restarts.
