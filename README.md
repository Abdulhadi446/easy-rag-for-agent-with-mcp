# rag/mcp — your documents, answerable by anything

![Demo: sign in, open a knowledge base, ask a question, get a cited answer](docs/demo.gif)

<p>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white">
  <img alt="Postgres + pgvector" src="https://img.shields.io/badge/Postgres-pgvector-4169E1?logo=postgresql&logoColor=white">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-streamable%20HTTP-3ddc97">
  <img alt="Tests" src="https://img.shields.io/badge/tests-49%20passing-3ddc97">
</p>

Upload PDFs, Markdown or text once. Then query them **in the browser** or let **any MCP-capable agent**
use them as a tool. Six chunking strategies, three retrieval modes, hybrid search with RRF,
optional reranking, and strict per-knowledge-base isolation — with defaults that work before you
open the settings page.

> **Local-first:** embeddings and reranking run on your CPU with ONNX. No API key is required to
> ingest or search. A chat model key is only needed for the web chat; MCP agents bring their own.

**Demo login:** `demo@example.com` / `demo-password` (created by `flask seed-demo`).
There is no hosted instance — run it locally in about two minutes.

[Watch the 32-second demo (MP4)](docs/demo.mp4) · [Screenshots](#screenshots) · [Quick start](#quick-start) · [MCP setup](#connect-an-agent-mcp) · [Source on GitHub](https://github.com/Abdulhadi446/easy-rag-for-agent-with-mcp)

---

## Table of contents

- [What you get](#what-you-get)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [How ingestion works](#how-ingestion-works)
- [How retrieval works](#how-retrieval-works)
- [Chunking strategies](#chunking-strategies)
- [Zero-config defaults](#zero-config-defaults)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Connect an agent (MCP)](#connect-an-agent-mcp)
- [Web chat](#web-chat)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Design decisions](#design-decisions)
- [Security model](#security-model)
- [Limitations and roadmap](#limitations-and-roadmap)

---

## What you get

| | |
|---|---|
| **Accounts and isolation** | Signup/login, CSRF protection, scrypt password hashing, per-user knowledge bases. Every lookup filters by owner and returns 404 for someone else's ids. |
| **Ingestion** | PDF (PyMuPDF, heading detection), Markdown (ATX + setext headings), TXT. Upload → parse → chunk → embed → ready, with live status in the UI. Duplicate detection by SHA-256, delete cascades to chunks, one-click re-index. |
| **Six chunkers** | Recursive (default), fixed size, sentence, sentence window, parent–child, semantic. Each is an isolated class behind one interface. |
| **Retrieval** | pgvector cosine + HNSW, Postgres full-text search with GIN, hybrid fusion via Reciprocal Rank Fusion, optional local cross-encoder or Cohere reranking, optional query rewriting. |
| **Web chat** | Streaming answers (SSE), Simple RAG and Agentic modes, expandable citations with filename, page and heading. |
| **Remote MCP server** | Streamable HTTP, bearer token per knowledge base, five tools designed for agentic RAG. Hashed tokens, rotation and revocation. |
| **Evidence** | 49 tests including tenant-isolation proofs, plus an evaluation script that reports hit@5 and MRR for every chunker × retrieval combination. |

---

## Screenshots

### Dashboard

![Knowledge base dashboard](docs/dashboard.png)

### Documents and ingestion status

![Documents table with ready status, page counts and chunk counts](docs/documents.png)

### Chat with citations

<p>
  <img alt="Chat answer with tool trace" src="docs/chat-answer.png" width="49%">
  <img alt="Expandable source list under the answer" src="docs/chat-citations.png" width="49%">
</p>

Answers stream token by token. Every answer carries the passages it used; click a source to read the
exact chunk, with filename, page and heading path.

### Advanced settings

![Per-knowledge-base chunking and retrieval settings](docs/settings.png)

Every option has a plain-language explanation, and **Reset to defaults** restores the zero-config
profile. Changing a chunking field automatically re-indexes the knowledge base's documents.

### Connect an agent

![MCP endpoint, copy-ready client config and token management](docs/tokens.png)

The dashboard shows a copy-ready MCP client configuration. Tokens are displayed once and stored only
as SHA-256 hashes.

---

## Architecture

One Flask process serves the UI and REST endpoints. Ingestion runs in background worker threads
(or a separate process). The MCP server is a second ASGI app so long-lived streamable HTTP sessions
never block Flask.

```mermaid
flowchart LR
    subgraph Browser
        UI[Web UI<br/>HTMX + vanilla JS]
    end

    subgraph Agent
        MCPC[MCP client<br/>Claude, Cursor, custom]
    end

    subgraph App["rag/mcp"]
        FL[Flask app factory]
        AUTH[Auth + CSRF]
        KB[Knowledge bases]
        DOC[Documents]
        CHAT[Chat SSE]
        WORK[Ingestion worker<br/>threads]
        MCPS[MCP server<br/>FastMCP / streamable HTTP]
        RET[Retrieval pipeline]
    end

    subgraph Data["State"]
        PG[(Postgres 17<br/>pgvector + tsvector)]
        FS[(Upload volume)]
        RD[(Redis<br/>optional)]
    end

    LLM[[OpenAI-compatible<br/>chat model]]
    EMB[[Local ONNX embeddings<br/>or OpenAI embeddings]]
    RR[[Local cross-encoder<br/>or Cohere]]

    UI --> FL
    AUTH --> FL
    FL --> KB & DOC & CHAT
    MCPC -->|Bearer token| MCPS
    CHAT --> RET
    MCPS --> RET
    RET --> PG
    RET -. rerank .-> RR
    CHAT --> LLM
    DOC --> WORK
    WORK --> FS
    WORK --> EMB
    WORK --> PG
    FL --> PG
    WORK -. queue .-> RD
```

MCP runs on `:8001` by default and Flask on `:8000`; both share the database and settings.

### Request flow: web chat

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant F as Flask /chat
    participant R as Retriever
    participant P as Postgres
    participant L as LLM

    U->>F: POST /chat/stream (question, mode)
    F->>R: retrieve(kb, question)
    R->>P: pgvector cosine + ts_rank
    P-->>R: candidate rows
    R->>R: RRF merge → dedupe parents → rerank
    R-->>F: top-k SearchHits
    F->>L: system prompt + numbered passages
    loop streaming
        L-->>F: token delta
        F-->>U: event: token
    end
    F-->>U: event: citations
    F-->>U: event: done
```

### Request flow: MCP agent

```mermaid
sequenceDiagram
    autonumber
    participant A as Agent
    participant M as MCP server
    participant D as Token lookup
    participant S as service.py
    participant P as Postgres

    A->>M: initialize + tools/list (Authorization: Bearer rag_…)
    M->>D: SHA-256(token) → api_tokens
    D-->>M: knowledge_base_id
    A->>M: tools/call search_documents
    M->>S: search(kb, query)
    S->>P: hybrid retrieval scoped to kb.id
    P-->>S: chunks + citations
    S-->>M: JSON results
    M-->>A: passages with filename + page
    A->>M: get_surrounding_context(chunk_id)
    M-->>A: neighbours in the same document
```

The knowledge base is bound to the request with a `ContextVar`; tool arguments can never select
another tenant's data.

---

## How ingestion works

```mermaid
flowchart TD
    UP[Upload<br/>PDF · MD · TXT] --> VAL{Validate}
    VAL -->|bad type / too big / no text| ERR[Reject with a clear message]
    VAL -->|ok| HASH[SHA-256 dedupe per KB]
    HASH --> STORE[(Upload volume)]
    STORE --> JOB[(jobs table<br/>queued)]
    JOB --> CLAIM[Worker claims<br/>FOR UPDATE SKIP LOCKED]
    CLAIM --> PARSE[Parser<br/>blocks + headings + pages]
    PARSE --> CHUNK[Chunker<br/>per-KB strategy]
    CHUNK --> EMBED[Embed non-parent chunks]
    EMBED --> SAVE[(documents + chunks<br/>vector + tsvector)]
    SAVE --> READY[status = ready]
    PARSE -->|scanned PDF / empty| FAIL[status = failed<br/>actionable error]
    EMBED -->|rate limit| RETRY[Exponential backoff<br/>max 3 attempts]
```

- **Status is first-class:** `queued → processing → ready | failed`, polled by the documents table.
- **Failures are visible and specific:** encrypted PDFs, scanned images with no text, empty files and
  unsupported types each produce a human-readable error, stored on the document.
- **Re-index is idempotent:** a job runs against a document id; old chunks are replaced in one
  transaction, and parent links are rebuilt.
- **Deleting a document** cascades to chunks and vectors at the database level and removes the stored
  file.
- **Duplicate uploads** are skipped unless the previous attempt failed, in which case the existing
  row is reused.

The worker is durable: jobs live in Postgres, are claimed with `SKIP LOCKED`, retried with a 5s → 10s
→ 20s backoff, and stuck `running` jobs are reset on worker start. Run it in-process
(`python -m app`) or separately (`python -m ingestion.worker`).

---

## How retrieval works

```mermaid
flowchart LR
    Q[Question] --> RW{Query rewrite?}
    RW -->|yes| LLM[LLM rewrites query]
    RW -->|no| EMB
    LLM --> EMB[Embed query]
    EMB --> V[Vector search<br/>HNSW cosine]
    Q --> K[Keyword search<br/>websearch_to_tsquery]
    V --> M{Mode}
    K --> M
    M -->|vector| D
    M -->|keyword| D
    M -->|hybrid| RRF[Reciprocal Rank Fusion<br/>k = 60]
    RRF --> D[Dedupe by parent]
    D --> RR{Rerank?}
    RR -->|yes| CE[Top 40 → cross-encoder → top 5]
    RR -->|no| TOP[Top k]
    CE --> TOP
    TOP --> OUT[Hits with citations]
```

- **Vector search** uses pgvector's cosine distance with an HNSW index (`m=16`,
  `ef_construction=64`). Scores are `1 - distance`.
- **Keyword search** uses Postgres full-text search (`ts_rank_cd`) over a generated `tsvector`
  column with a GIN index, with a `plainto_tsquery` fallback for punctuation-heavy questions.
- **Hybrid** merges both ranked lists with RRF, so a passage found by both retrievers rises to the
  top without calibrating incomparable scores.
- **Parent–child** results are deduplicated by `parent_id`, returning the larger parent text once.
- **Reranking** fetches a wider candidate pool, scores `(query, passage)` pairs with a local
  cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) or Cohere Rerank v2, and keeps the best `top_k`.
  If the model cannot load, retrieval degrades gracefully to the fused order.
- **Every hit carries** chunk id, document id, filename, page, heading path, score and citation.
  Sentence-window chunks return their window; parent–child chunks return the parent.

---

## Chunking strategies

Each strategy is a separate class implementing `Chunker.split(blocks) -> list[ChunkDraft]`, registered
in `ingestion/chunkers/__init__.py`, so adding one is a two-file change.

| Strategy | What it does | Best for |
|---|---|---|
| **Recursive** (default) | Splits on headings, then paragraphs, then sentences; ~500 tokens with ~50 overlap. Falls back to paragraphs/sentences when a PDF has no headings. | Almost everything. |
| **Fixed size** | Cuts every N tokens with configurable overlap, word-aligned. | Uniform pricing tables, logs, transcripts. |
| **Sentence** | One sentence per chunk, truncated if longer than the configured size. | Precise fact lookup. |
| **Sentence window** | Embeds single sentences, stores N sentences before/after, returns the window. | Fine-grained matching with readable context. |
| **Parent–child** | Embeds ~200-token children; each points at a ~1000-token parent. Only parents are returned, deduplicated. | Dense technical docs where precision and context both matter. |
| **Semantic** (optional) | Embeds sentences and starts a new chunk when cosine similarity to the previous sentence drops below a threshold. | Long prose with drifting topics. |

Chunk records store text, embedding, document id, knowledge base id, page, heading path, index,
token count, optional parent id / window text, and a generated `tsvector` for keyword search.

---

## Zero-config defaults

A new account gets a starter knowledge base with settings that already work:

| Setting | Default | Why |
|---|---|---|
| Chunking | `recursive`, 500 tokens, 50 overlap | Structure-aware, safe for PDFs and Markdown |
| Retrieval | `hybrid` | Handles both paraphrases and exact identifiers |
| Candidates | 40 | Enough for reranking without latency spikes |
| Rerank | on when a provider is available | Big precision win on ambiguous queries |
| Top results | 5 | Fits comfortably in a chat prompt |
| Query rewrite | off | Extra latency only helps vague questions |
| Embeddings | local ONNX `bge-small-en-v1.5`, 384d | No key, no per-token cost, offline after first download |

All of it is overridable per knowledge base under **Settings**, with a **Reset to defaults** button.

---

## Quick start

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`)
- Docker with Compose (for Postgres + pgvector)
- ~500 MB disk for the local models (downloaded on first use)

### 1. Clone and install

```bash
git clone https://github.com/Abdulhadi446/easy-rag-for-agent-with-mcp.git
cd easy-rag-for-agent-with-mcp

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SECRET_KEY
```

Set `OPENROUTER_API_KEY` (or `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL`) to enable web-chat answers.
Ingestion, search and MCP work without it.

### 3. Start Postgres with pgvector

```bash
docker compose up -d
```

This starts `pgvector/pgvector:pg17` on `:5432` and Redis on `:6379`. The default credentials are
`rag` / `rag` / `rag`, matching `.env.example`.

### 4. Create the schema

```bash
alembic upgrade head
```

### 5. Run

```bash
python -m app
```

- Web UI: <http://127.0.0.1:8000>
- MCP endpoint: <http://127.0.0.1:8001/mcp>

The command starts the ingestion worker and the MCP server alongside Flask. For separate processes:

```bash
python -m ingestion.worker    # ingestion only
python -m mcp_server          # MCP only
python -m app --no-worker --no-mcp
```

### 6. Try it

```bash
flask --app wsgi seed-demo
flask --app wsgi create-user you@example.com   # prompts for a password
```

Log in as `demo@example.com` / `demo-password`, open **Demo library**, upload
[`samples/northwind-policies.md`](samples/northwind-policies.md), wait for **ready**, then ask
*“How long do EU customers have to withdraw?”*

---

## Configuration

All settings are environment variables, read once through `app/config.py`. See `.env.example` for the
annotated list.

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | dev value | Flask session signing — change it |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | Postgres + pgvector |
| `UPLOAD_DIR` | `./uploads` | Stored originals, outside the web root |
| `MAX_UPLOAD_MB` | `25` | Per-file limit |
| `EMBEDDINGS_PROVIDER` | `local` | `local`, `openai`, or `hash` (tests only) |
| `EMBEDDING_MODEL` | `Xenova/bge-small-en-v1.5` | Hugging Face ONNX repo or API model |
| `EMBEDDING_DIM` | `384` | Must match the vector column (migration) |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | — | For OpenAI-compatible embeddings |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat endpoint |
| `LLM_MODEL` | `openrouter/free` | Chat model used by the web UI |
| `OPENROUTER_API_KEY` / `LLM_API_KEY` | — | Chat credentials |
| `RERANK_PROVIDER` | `local` | `local`, `cohere`, or `none` |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder repo |
| `COHERE_API_KEY` | — | When `RERANK_PROVIDER=cohere` |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8001` | MCP bind address |
| `MCP_PUBLIC_URL` | `http://localhost:8001/mcp` | URL shown in the dashboard |
| `WORKER_ENABLED` | `1` | Start ingestion threads with `python -m app` |
| `WORKER_CONCURRENCY` | `2` | Worker threads |
| `LLM_MAX_TOKENS` | `1200` | Answer budget |
| `AGENTIC_MAX_TOOL_CALLS` | `5` | Agentic chat tool budget |

Changing `EMBEDDING_DIM` requires a matching Alembic migration and re-embedding. The app refuses to
start if the model dimension and the database column disagree.

### Docker Compose services

```mermaid
flowchart LR
    subgraph compose["docker compose"]
        DB[("db<br/>pgvector/pgvector:pg17<br/>:5432")]
        REDIS[("redis<br/>redis:7-alpine<br/>:6379")]
    end
    APP[rag/mcp process] --> DB
    APP -. optional queue .-> REDIS
    DB --- VOL[(pgdata volume)]
    REDIS --- RVOL[(redisdata volume)]
```

---

## Connect an agent (MCP)

1. Open a knowledge base → **Connect an agent** → **Create token**.
2. Copy the token immediately; it is shown once.
3. Paste the generated configuration into your MCP client:

```json
{
  "mcpServers": {
    "demo-library": {
      "type": "http",
      "url": "http://localhost:8001/mcp",
      "headers": {
        "Authorization": "Bearer rag_your_token_here"
      }
    }
  }
}
```

The server speaks **streamable HTTP** and authenticates every request with the bearer token. Tokens
are scoped to one knowledge base; rotate or revoke them from the same page.

### Tools

| Tool | Arguments | Returns |
|---|---|---|
| `search_documents` | `query`, `top_k=5`, `mode=None` | Cited passages; `mode` overrides `vector`, `keyword`, `hybrid`. |
| `list_documents` | — | Documents with status, pages, chunks, upload time. |
| `get_document_chunk` | `chunk_id` | Full chunk text, parent text when present, citation. |
| `get_surrounding_context` | `chunk_id`, `before=2`, `after=2` | Neighbouring chunks in the same document. |
| `get_document_outline` | `document_id` | Heading tree with page numbers where available. |

The tool descriptions are written for the model, not for humans: they explain when to search again,
when to drill into a hit, and to cite filename and page. That enables agentic RAG — the agent can
search, rephrase, expand and navigate before answering.

### Example agent prompt

```text
You are researching our internal policy library. Use search_documents with several
different phrasings, then get_surrounding_context for promising hits. Answer only from
the returned passages and cite "filename, page N" for every claim. If the library does
not contain the answer, say so.
```

### Authentication flow

```mermaid
flowchart TD
    REQ[HTTP request] --> H{Authorization header?}
    H -->|missing| U[401 + WWW-Authenticate]
    H -->|Bearer rag_…| SHA[SHA-256 token]
    SHA --> DB{api_tokens<br/>revoked_at IS NULL<br/>user active}
    DB -->|no match| U
    DB -->|match| CTX[ContextVar = knowledge_base_id]
    CTX --> TOOL[Tool executes scoped queries]
    TOOL --> AUDIT[Update last_used_at]
```

No plaintext token is ever stored or logged. The middleware runs before MCP sees the request, so
unauthenticated requests cannot create sessions or enumerate tools.

---

## Web chat

- **Simple RAG** — retrieve once, put numbered passages in the prompt, stream the answer. Fast and
  predictable.
- **Agentic** — give the model the same five search tools, let it call them up to
  `AGENTIC_MAX_TOOL_CALLS` times, then stream a final answer. Tool calls appear as chips above the
  answer.

Both modes share the retrieval pipeline with MCP; there is no separate "chat search" path. Citations
are streamed as a final SSE event and rendered as expandable `<details>` elements containing the
source text.

The system prompt instructs the model to answer **only** from retrieved context and to say clearly
when the documents do not cover the question.

---

## Evaluation

`evaluation/` measures retrieval quality instead of guessing. It creates a throwaway user, ingests the
sample documents with each chunker, asks every question in
[`evaluation/dataset.json`](evaluation/dataset.json), and reports hit@5 plus MRR for each retrieval
mode.

```bash
python -m evaluation --docs samples --dataset evaluation/dataset.json
python -m evaluation --docs samples --rerank          # include the cross-encoder
python -m evaluation --json-out eval-report.json      # machine-readable
```

Result on the bundled dataset (14 questions, 3 documents, local `bge-small` embeddings, no rerank):

```text
Hit@5 — northwind-aurora-eval (14 questions)

strategy            vector   keyword    hybrid
----------------------------------------------
recursive             100%       29%      100%
fixed                 100%       43%      100%
sentence               93%       21%       93%
sentence_window       100%       21%      100%
parent_child          100%       29%      100%
semantic               93%       21%       93%

MRR (mean reciprocal rank)

strategy            vector   keyword    hybrid
----------------------------------------------
recursive             0.77      0.29      0.81
fixed                 0.86      0.43      0.96
sentence              0.79      0.21      0.79
sentence_window       1.00      0.21      1.00
parent_child          0.85      0.29      0.66
semantic              0.79      0.21      0.79
```

**How to read it.** The questions are paraphrases, so keyword-only search misses vocabulary mismatches
(“hardware warranty” vs. “two-year warranty”) — exactly the gap hybrid search exists to close. Vector
and hybrid both hit 100% for structure-aware chunkers. Sentence and semantic chunking split short
facts from their headings, which costs a little recall and explains the 93%. MRR separates the
strategies that put the right passage first: sentence-window is perfect here because one sentence
*is* the answer, while parent–child sometimes ranks a broader parent above the precise child.

These numbers are a starting point, not a leaderboard: replace the dataset with your own questions
and expected passages before choosing settings.

---

## Tests

```bash
uv run pytest -q          # 49 passed
uv run ruff check .
```

| Area | What is covered |
|---|---|
| Auth and security | Signup/login/logout, duplicate emails, normalization, open-redirect rejection, CSRF enforcement, throttle, salted hashes, token randomness and hashing |
| Parsers | Markdown headings and frontmatter, text paragraphs, PDF heading detection via font size, scanned-PDF error, empty input |
| Chunkers | Sizes, overlap, heading boundaries, parent–child links, sentence windows, semantic splitting, unknown strategy |
| Pipeline | Upload validation, duplicate handling, parse → embed → ready, empty-file failure, delete cascade, ownership checks |
| Retrieval | RRF ordering and fusion, parent deduplication, cited vector hits, **two-user isolation** (foreign KBs return 404 and retrieve nothing) |
| MCP | Token shown once and stored hashed, revoke/rotate, 401 for missing/bad bearer tokens, service-level scoping |
| Chat | SSE error handling, simple mode tokens + citations, agentic tool loop, fallback without a model |
| Settings | Defaults render, validation errors, reset restores defaults |

The isolation test is deliberately adversarial: user B's empty knowledge base is queried with the
same question that retrieves user A's chunk, then every direct URL to A's data is requested.

---

## Project layout

```text
.
├── app/
│   ├── __init__.py            # app factory, blueprint registration, error handlers
│   ├── config.py              # env-backed Settings + DEFAULT_KB_SETTINGS
│   ├── models.py              # SQLAlchemy models, vector/tsvector columns, indexes
│   ├── db.py                  # engine, scoped session, context manager
│   ├── auth/                  # signup, login, logout, login_required
│   ├── knowledge_bases/       # dashboard, KB page, settings routes
│   ├── documents/             # upload, list fragment, detail, delete, reprocess
│   ├── chat/                  # SSE chat: simple + agentic
│   ├── tokens/                # MCP token create/rotate/revoke
│   ├── templates/             # Jinja: auth, dashboard, documents, chat, settings, tokens
│   └── static/app.css         # custom dark design system, no framework
├── ingestion/
│   ├── parsers.py             # PyMuPDF + Markdown + text parsers
│   ├── tokens.py              # tiktoken helpers, sentence splitting
│   ├── pipeline.py            # durable jobs, parse → chunk → embed → store
│   ├── worker.py              # threaded worker with app context
│   └── chunkers/              # six strategies behind one interface
├── retrieval/
│   ├── vector.py              # pgvector cosine + HNSW
│   ├── keyword.py             # Postgres FTS + ts_rank_cd
│   ├── hybrid.py              # Reciprocal Rank Fusion
│   ├── rerank.py              # local cross-encoder / Cohere
│   ├── rewrite.py             # optional query rewriting
│   └── pipeline.py            # retrieve(), mode and rerank orchestration
├── mcp_server/
│   ├── server.py              # FastMCP streamable HTTP + bearer middleware
│   └── service.py             # shared, tenant-scoped tool implementations
├── embeddings.py              # local ONNX / OpenAI / hash providers
├── llm.py                     # OpenAI-compatible chat + streaming + tool calls
├── evaluation/                # hit@5 / MRR harness and dataset
├── samples/                   # three demo documents
├── tests/                     # 49 tests
├── alembic/                   # initial migration (pgvector + HNSW + GIN)
├── docker-compose.yml
├── .env.example
└── wsgi.py
```

---

## Design decisions

**Local embeddings by default.** `bge-small-en-v1.5` runs on CPU through ONNX at ~35 MB quantized and
performs well on retrieval benchmarks. It removes the API key, the per-token cost and the network
round-trip from ingestion; OpenAI embeddings are a single env-var swap.

**Postgres for everything.** Vectors, full-text index, jobs and tenant data live in one transactional
store. There is no second service to keep consistent, and deletion cascades guarantee no orphaned
vectors.

**RRF instead of score blending.** Cosine similarity and `ts_rank_cd` are not comparable. Reciprocal
Rank Fusion only needs ranks, so hybrid search works without tuning weights per corpus.

**Durable jobs, not fire-and-forget threads.** A crash mid-ingestion leaves a `queued` job that another
worker can claim. `SKIP LOCKED` makes multiple workers safe without a broker; Redis is available for a
future RQ/Celery swap.

**One retrieval pipeline.** The MCP server, web chat and evaluation harness all call
`retrieval.retrieve()`. A fix or setting applies everywhere, and the evaluation numbers describe
exactly what agents get.

**Tenant isolation in the query, not the template.** Helpers such as
`get_knowledge_base_or_404()` always filter by `user_id`; MCP tools resolve the knowledge base from
the token, never from tool arguments. Foreign ids return 404 rather than 403, so existence is not
leaked.

**No frontend framework.** Server-rendered Jinja, a small polling fragment for ingestion status, and
vanilla JS for SSE keep the surface small and the page fast.

---

## Security model

- **Passwords:** scrypt via Werkzeug, minimum 8 characters, no surrounding whitespace.
- **Sessions:** HTTP-only, `SameSite=Lax`, 14-day lifetime; login rate-limited per email + IP.
- **CSRF:** every unsafe request requires a session-bound token, either as a form field or
  `X-CSRF-Token` header. The MCP blueprint is exempt because it uses bearer auth.
- **API tokens:** 256 bits of entropy, `rag_` prefix, SHA-256 hash at rest, prefix for display, rotation
  and revocation, `last_used_at` audit.
- **Uploads:** extension allowlist, PDF magic-byte check, binary/NUL rejection, per-file and total size
  limits, randomized storage names, path traversal protection, files outside the static root.
- **Logging:** document contents and tokens are never logged; provider errors are summarized.
- **SQL:** SQLAlchemy Core/ORM throughout; the only raw SQL is the generated `tsvector` expression and
  migration DDL.
- **MCP:** authentication happens in ASGI middleware before the MCP app; tool calls execute inside a
  knowledge-base context variable; DNS-rebinding protection is disabled only because bearer tokens,
  not network origin, are the security boundary (put it behind TLS in production).

For a public deployment: set `SESSION_COOKIE_SECURE=True`, terminate TLS in front of both ports, set a
strong `SECRET_KEY`, move the worker to its own process, and restrict Postgres/Redis to the compose
network.

---

## Limitations and roadmap

Honest edges of this version:

- **No OCR.** Scanned PDFs fail with a clear message instead of silently producing nothing. Wiring
  Docling or Tesseract is the next parser step.
- **Chat history is per browser tab.** Conversations stream but are not persisted server-side; the MCP
  server is stateless by design.
- **One embedding dimension per database.** Changing models requires a migration and re-embedding. A
  per-knowledge-base dimension column is future work.
- **No teams or sharing.** Knowledge bases belong to one user; tokens are the only delegation
  mechanism.
- **In-process worker by default.** Fine for a single node; for multiple web replicas, run
  `python -m ingestion.worker` separately (jobs are already safe under `SKIP LOCKED`).
- **No per-token rate limits.** Add a reverse-proxy limit or a token bucket before exposing MCP
  publicly.
- **Redis is provisioned but unused.** It is there for the RQ/Celery upgrade, not required today.

Planned next: OCR fallback, persisted chat sessions, streaming tool events inside MCP, per-KB
embedding models, and a Docker image for one-command deployment.

---

<p align="center">
  <sub>Built with Flask, Postgres + pgvector, FastMCP, PyMuPDF, tiktoken and ONNX Runtime.</sub><br>
  <sub>If this saved you from wiring RAG and MCP together by hand, star the repo.</sub>
</p>
