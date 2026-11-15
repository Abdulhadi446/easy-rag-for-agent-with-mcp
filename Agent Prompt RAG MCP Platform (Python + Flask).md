# Agent Prompt: RAG MCP Platform (Python + Flask)

Paste everything below into your coding agent.

---

## Goal

Build a web platform where users upload documents (PDF, Markdown, TXT), and then query them either through a built-in web chat or by connecting their own AI agent to a remote MCP server. The system must work with zero configuration (smart defaults) but let advanced users customize chunking and retrieval per knowledge base.

## Tech stack

- Python 3.11+, Flask for the web app and REST API
- PostgreSQL with the pgvector extension for vectors, plus Postgres full-text search (tsvector, GIN index) for keyword search
- SQLAlchemy and Alembic for models and migrations
- Official MCP Python SDK (FastMCP) mounted as a remote MCP server over streamable HTTP, served alongside Flask
- Background jobs for ingestion: Celery or RQ with Redis (a simple thread worker is acceptable for the first version)
- AI provider calls must go through one abstraction layer (`llm.py` and `embeddings.py`) so the provider can be swapped by environment variable. Default to OpenAI-compatible APIs: `text-embedding-3-small` for embeddings and a chat model for the web chat. Optionally support Voyage or Cohere for reranking.
- Document parsing: PyMuPDF or Docling for PDFs (keep heading and structure information when possible), plain parsing for Markdown and TXT
- Frontend: simple server-rendered templates with HTMX or light JavaScript. No heavy framework needed.

## Core features

### 1. Accounts and API keys

- Signup, login, sessions for the web app
- Every user can create multiple **knowledge bases**
- Every knowledge base has its own MCP access token (generate, revoke, rotate). Store only hashed tokens.
- All queries must be strictly scoped to the owner's data. Add tests proving one user can never retrieve another user's chunks.

### 2. Document ingestion

- Upload PDF, MD, TXT with size limits and file type validation
- Pipeline: upload, parse, clean, chunk, embed, store, with a status field (queued, processing, ready, failed) visible in the UI
- Store document metadata: filename, page numbers, section headings, upload time
- Deleting a document deletes all its chunks and vectors
- Re-ingestion when the user changes chunking settings

### 3. Chunking strategies (selectable per knowledge base)

Implement each as a separate class with a common interface so new ones can be added easily.

1. **Recursive (default)**: split by headings first, then paragraphs, then sentences, enforcing a max size of about 500 tokens with about 50 tokens of overlap. For PDFs with no detectable headings, fall back to paragraph and sentence splitting.
2. **Fixed size**: configurable token size and overlap.
3. **Sentence**: one sentence per chunk.
4. **Sentence window**: embed and search single sentences, but store the surrounding N sentences (default 3 before and 3 after) and return that window to the caller.
5. **Parent-child**: split into large parent chunks (about 1000 tokens, or a full section) and small child chunks (about 200 tokens). Only embed children. Each child stores a parent id. At retrieval time, return the parent and deduplicate when several children share one parent.
6. **Semantic (optional)**: embed sentences and start a new chunk when similarity to the previous sentence drops below a threshold.

Every chunk stores: text, embedding, document id, knowledge base id, page number, heading path, chunk index, and optional parent id or window text.

### 4. Retrieval modes (selectable per knowledge base and overridable per query)

1. **Vector search**: cosine similarity with pgvector (HNSW index).
2. **Keyword search**: Postgres full-text search ranked with `ts_rank`.
3. **Hybrid (default)**: run vector and keyword search, merge with Reciprocal Rank Fusion.
4. **Reranking (optional toggle, on by default if a rerank key exists)**: retrieve the top 30 to 50 candidates, rerank with a cross-encoder or a rerank API, and return the top 5.
5. Optional **query rewriting**: use the LLM to rewrite vague questions into better search queries before retrieval.

All results must return: chunk text (or parent or window text where relevant), source filename, page number, heading path, and a relevance score, so answers can cite sources.

### 5. Default settings (zero-config experience)

A user who just uploads a PDF and starts chatting must get good results with no setup. Defaults: recursive chunking, hybrid search, reranking on if available, top 5 results. Put all other options under an "Advanced settings" section per knowledge base, with a short plain-language explanation next to each option, plus a "Reset to defaults" button.

### 6. MCP server

Expose a remote MCP server (streamable HTTP) authenticated by the knowledge base token in an Authorization bearer header. Tools:

- `search_documents(query, top_k=5, mode=None)`: runs the knowledge base's configured retrieval and returns chunks with citations. `mode` optionally overrides vector, keyword, or hybrid.
- `list_documents()`: returns the documents in the knowledge base with status and page counts.
- `get_document_chunk(chunk_id)` and `get_surrounding_context(chunk_id, before, after)`: let an agent expand around a result.
- `get_document_outline(document_id)`: returns headings so an agent can navigate.

Write clear tool descriptions, because the connected agent reads them to decide how to use the tools. This enables agentic RAG: the agent can search multiple times, rephrase, and drill into documents. Show users a ready-to-copy connection snippet (URL plus token) in the dashboard.

### 7. Built-in web chat

- Chat page per knowledge base, with streaming responses
- The chat uses the same retrieval pipeline as the MCP server
- Two modes: **Simple RAG** (retrieve once, then answer) and **Agentic** (the LLM is given the search tools and can call them repeatedly, with a max of about 5 tool calls)
- Show citations under each answer (filename, page), expandable to the source chunk
- A system prompt that tells the model to answer only from retrieved context and to say so when the documents do not contain the answer

## Project structure

Suggest and follow a clean layout: `app/` (Flask factory, blueprints for auth, knowledge bases, documents, chat), `ingestion/` (parsers, chunkers), `retrieval/` (vector, keyword, hybrid, rerank), `mcp_server/`, `llm.py`, `embeddings.py`, `models.py`, `tests/`, `docker-compose.yml` (Postgres with pgvector, Redis), `.env.example`, and a README with setup steps.

## Quality requirements

- Write tests for each chunker (sizes, overlap, parent-child linking, sentence windows) and for retrieval (RRF merging, user isolation).
- Add a small evaluation script: given a set of test questions and expected source passages, report hit rate at top 5 for each chunking and retrieval combination, so settings can be compared with evidence.
- Handle errors gracefully: failed parses, empty PDFs (scanned images with no text should show a clear message), API rate limits with retries.
- Never log document contents or tokens. Validate all uploads.

## Build order

Work in milestones and pause for my review after each:

1. Project skeleton, Docker Compose, database models, auth
2. Upload and ingestion with the default recursive chunker and vector search
3. Hybrid search and reranking
4. Remaining chunkers (sentence, sentence window, parent-child, semantic) and the advanced settings UI
5. MCP server with token auth and all tools
6. Web chat with simple and agentic modes, plus citations
7. Tests, evaluation script, README polish

Before writing code, briefly present your plan and any questions, then start with milestone 1.
