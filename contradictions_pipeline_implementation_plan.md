# Contradiction Detection Pipeline — Full Implementation Plan
**For: GPT-5.2 Codex**
**Note to Codex**: If anything below is ambiguous or you need a design decision not covered here, stop and ask the user before writing code. Do not make assumptions on data paths, model names, or schema fields.

---

## Project Overview

Build an automated pipeline that detects and categorises contradictions between computer science research papers. The pipeline ingests precomputed retrieval vectors plus their metadata, clusters papers by research topic, identifies semantically similar pairs, extracts claims, retrieves evidence, and runs an ensemble NLI system to detect contradictions.

### Data Inputs (fixed)
- Embeddings: `project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/data/processed/retrieval_vectors.npy`
- Metadata: **required** mapping file that aligns rows in `retrieval_vectors.npy` to paper IDs, titles, abstracts, and sectioned text. If not already present, the user must provide its path and schema before implementation.

---

## Deployment Context — HPC and Local

This pipeline is designed to run on an HPC cluster using **SLURM** as the job scheduler and **Singularity** as the container runtime. It must also be fully reproducible locally — anyone must be able to `git clone` the repo and run it on their own machine using Docker or a plain Python virtualenv. The two modes share identical pipeline code. All HPC-specific logic is isolated in `hpc/`.

### Design rules Codex must follow

- Never hardcode absolute paths anywhere in `pipeline/`, `workers/`, `db/`, or `models/`. All paths come from `.env` or CLI arguments.
- All entry-point scripts must accept a `--env` argument pointing to a `.env` file so local, HPC, and CI environments each have their own config without touching source code. Example: `python scripts/run_pipeline.py --env hpc/env/hpc.env`
- The Singularity container bundles all Python dependencies. No `pip install` at runtime on HPC nodes.
- Redis cannot run as a persistent daemon on a compute node. On HPC, Celery is replaced with a direct sequential loop — see the HPC section below.
- All file writes (SQLite DB, logs, outputs) must go to paths defined in `.env`, not relative to the repo root, because on HPC the working directory is typically `$SCRATCH` or `$WORK`, not the repo.

---

## Repository Structure

```
contradiction_pipeline/
├── config/
│   └── settings.py               # All config constants loaded from .env
├── data/
│   └── README.md                 # Instructions for placing UnarXive data
├── pipeline/
│   ├── __init__.py
│   ├── stage1_clustering.py      # CSO tagging + lexical filter + cluster formation
│   ├── stage2_anchor.py          # SPECTER 2 embedding + anchor selection + pairing
│   ├── stage3_similarity.py      # Cosine similarity filter + pair list generation
│   ├── stage4_claims.py          # Phi-3 claim extraction
│   ├── stage5_evidence.py        # Llama evidence retrieval
│   ├── stage6_nli.py             # Ensemble NLI (Bloomz + Llama)
│   └── stage7_typing.py          # Contradiction typing + explanation
├── workers/
│   ├── celery_app.py             # Celery + Redis config (local mode only)
│   └── tasks.py                  # Celery task definitions (local mode only)
├── db/
│   └── schema.py                 # SQLite schema + helper functions
├── models/
│   └── vllm_client.py            # Async HTTP client for vLLM endpoints
├── prompts/
│   ├── claim_extraction.txt
│   ├── evidence_retrieval.txt
│   ├── nli_zero_shot.txt
│   └── typing_fewshot.txt
├── evaluation/                   # OPTIONAL — only active if GOLD_STANDARD_PATH is set
│   ├── gold_standard.py
│   ├── metrics.py
│   └── README.md
├── scripts/
│   ├── run_pipeline.py           # Main entry point (local: Celery, HPC: sequential loop)
│   ├── run_evaluation.py         # OPTIONAL evaluation runner
│   ├── generate_report.py        # Generates self-contained report.html from SQLite
│   └── resume_cluster.py         # Resume a specific cluster by ID
├── hpc/
│   ├── container/
│   │   ├── Dockerfile            # Defines the image (built locally, converted to .sif on HPC)
│   │   └── build_container.sh    # Commands to build the Singularity .sif from the Dockerfile
│   ├── jobs/
│   │   ├── 01_stage1.sh          # SLURM job: ingest + cluster (CPU, high RAM)
│   │   ├── 02_embed.sh           # SLURM job: SPECTER 2 embeddings (GPU)
│   │   ├── 03_models.sh          # SLURM job: launch vLLM model servers (GPU, long-running)
│   │   ├── 04_pipeline.sh        # SLURM job: stages 4-7 over all clusters (GPU)
│   │   └── 05_report.sh          # SLURM job: generate report.html (CPU, runs after pipeline)
│   └── env/
│       ├── local.env             # Example .env for local development
│       └── hpc.env               # Example .env for HPC ($SCRATCH paths, node hostnames)
├── Dockerfile                    # Symlink or copy of hpc/container/Dockerfile
├── tests/
│   └── test_each_stage.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component | Library/Tool | Version |
|---|---|---|
| Python | Python | 3.11 |
| Container (local) | Docker | latest |
| Container (HPC) | Singularity | 3.x |
| Job scheduler (HPC) | SLURM | cluster-dependent |
| Job queue (local) | Celery | 5.3.x |
| Message broker (local) | Redis | 7.x (via redis-py) |
| Checkpoint DB | SQLite | via Python stdlib sqlite3 |
| CSO classifier | cso-classifier | latest pip |
| Embeddings | sentence-transformers (SPECTER 2) | latest pip |
| LLM inference | vLLM HTTP API | self-hosted, 3 ports |
| HTTP client | httpx | async |
| Data loading | numpy + metadata loader | metadata file format to be confirmed |
| NLI (Bloomz) | via vLLM endpoint | |
| Config | python-dotenv | |
| Logging | Python logging + rotating file handler | |

---

## Configuration — `config/settings.py`

```python
# All values here are defaults. User should override via .env file.

RETRIEVAL_VECTORS_PATH = ""   # Required: path to retrieval_vectors.npy
METADATA_PATH = ""            # Required: path to metadata that maps vectors to papers
SQLITE_DB_PATH = "pipeline_state.db"
REDIS_URL = "redis://localhost:6379/0"
LOG_PATH = "logs/pipeline.log"

# CSO
CSO_MIN_CLUSTER_SIZE = 20
LEXICAL_DICT_PATH = ""        # Optional. Set to a file path to enable lexical filtering.
                              # If empty string or file does not exist, lexical filter is skipped.

# Embeddings
EMBEDDING_DIM = 768

# Similarity
SIMILARITY_THRESHOLD = 0.75

# vLLM endpoints — ask user for actual ports/hostnames
PHI3_ENDPOINT   = "http://localhost:8001/v1/completions"
BLOOMZ_ENDPOINT = "http://localhost:8002/v1/completions"
LLAMA_ENDPOINT  = "http://localhost:8003/v1/completions"

PHI3_MODEL_NAME   = "microsoft/Phi-3-mini-4k-instruct"
BLOOMZ_MODEL_NAME = "bigscience/bloomz-3b"       # NLI variant; confirm with user
LLAMA_MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# NLI
NLI_CONFIDENCE_THRESHOLD = 0.65   # Llama logprob threshold
ENSEMBLE_AGREE_ONLY = True         # If True, only emit contradiction when both agree

# Celery
CELERY_TASK_RETRIES = 3
CELERY_RETRY_BACKOFF = 60          # seconds

# Claim extraction
MAX_CLAIMS_PER_PAPER = 15
```

---

## Database Schema — `db/schema.py`

Create the following SQLite tables on first run. Use `IF NOT EXISTS` everywhere.

```sql
-- Tracks which clusters have been fully processed
CREATE TABLE IF NOT EXISTS cluster_progress (
    cluster_id       TEXT PRIMARY KEY,   -- CSO leaf term (e.g. "transformer_models")
    status           TEXT NOT NULL,       -- "pending", "stage1", "stage2", ..., "done", "error"
    paper_count      INTEGER,
    pair_count       INTEGER,
    error_message    TEXT,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per paper
CREATE TABLE IF NOT EXISTS papers (
    paper_id         TEXT PRIMARY KEY,   -- arXiv ID
    title            TEXT,
    abstract         TEXT,
    intro_text       TEXT,               -- first paragraph of introduction
    results_text     TEXT,
    conclusion_text  TEXT,
    limitations_text TEXT,
    future_work_text TEXT,
    pub_date         DATE,
    authors          TEXT,               -- JSON array of author strings
    cso_tags         TEXT,               -- JSON array of assigned CSO leaf terms
    embedding        BLOB                -- 768-dim float32, stored as numpy tobytes()
);

-- One row per cluster membership (many-to-many papers <-> clusters)
CREATE TABLE IF NOT EXISTS cluster_membership (
    cluster_id  TEXT,
    paper_id    TEXT,
    PRIMARY KEY (cluster_id, paper_id)
);

-- One row per anchor-challenger pair
CREATE TABLE IF NOT EXISTS candidate_pairs (
    pair_id          TEXT PRIMARY KEY,   -- UUID
    cluster_id       TEXT,
    anchor_id        TEXT,
    challenger_id    TEXT,
    similarity_score REAL,
    status           TEXT DEFAULT "pending",  -- "pending", "claims_extracted", "evidence_retrieved", "nli_done", "typed", "skipped"
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claims extracted from anchor papers
CREATE TABLE IF NOT EXISTS claims (
    claim_id    TEXT PRIMARY KEY,   -- UUID
    pair_id     TEXT,
    paper_id    TEXT,
    claim_text  TEXT,
    source_section TEXT,            -- "introduction", "results", "conclusion"
    claim_index INTEGER             -- order within the paper
);

-- Evidence chunks retrieved from challenger papers
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id   TEXT PRIMARY KEY,  -- UUID
    claim_id      TEXT,
    pair_id       TEXT,
    evidence_text TEXT,              -- raw passage, not summarised
    source_section TEXT
);

-- NLI results per claim-evidence pair
CREATE TABLE IF NOT EXISTS nli_results (
    nli_id           TEXT PRIMARY KEY,
    claim_id         TEXT,
    evidence_id      TEXT,
    pair_id          TEXT,
    bloomz_label     TEXT,           -- "entailment", "contradiction", "neutral"
    bloomz_score     REAL,
    llama_label      TEXT,
    llama_confidence REAL,           -- logprob-derived
    ensemble_label   TEXT,           -- "contradiction", "support", "neutral", "flagged"
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Final typed contradictions
CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id  TEXT PRIMARY KEY,
    pair_id           TEXT,
    claim_id          TEXT,
    evidence_id       TEXT,
    contradiction_type TEXT,          -- one of 5 taxonomy types
    explanation       TEXT,
    reviewer_label    TEXT,           -- Qwen2.5 review label; nullable
    reviewer_notes    TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Implement these helper functions in `db/schema.py`:
- `init_db(db_path)` — creates all tables
- `get_cluster_status(cluster_id)` — returns current status string
- `set_cluster_status(cluster_id, status, **kwargs)` — updates progress row
- `upsert_paper(paper_dict)` — inserts or replaces a paper row
- `get_papers_for_cluster(cluster_id)` — returns list of paper dicts
- `save_embedding(paper_id, embedding_array)` — stores numpy array as BLOB
- `load_embedding(paper_id)` — returns numpy float32 array
- `save_pair(pair_dict)` — inserts candidate pair
- `get_pending_pairs(cluster_id)` — returns pairs with status "pending"
- `update_pair_status(pair_id, status)` — updates pair status
- `save_claims(claim_list)` — bulk insert
- `save_evidence(evidence_list)` — bulk insert
- `save_nli_result(nli_dict)` — insert NLI row
- `save_contradiction(contradiction_dict)` — insert contradiction row

---

## Stage 1 — Ingestion and Clustering (`stage1_clustering.py`)

### Input
Precomputed embeddings from `RETRIEVAL_VECTORS_PATH` and a metadata file from `METADATA_PATH` that aligns each embedding row to a paper record.

### Paper fields to extract per record
```python
{
    "paper_id": str,           # arXiv ID
    "title": str,
    "abstract": str,
    "pub_date": str,           # ISO date string
    "authors": list[str],
    "body_text": list[dict],   # list of {heading: str, text: str}
    "cso_tags": list[str] | None,
}
```

### Section extraction logic
From `body_text`, extract sections by matching heading keywords (case-insensitive):
- `intro_text`: heading contains "introduction" — take first paragraph only (first 500 chars)
- `results_text`: heading contains "result" or "experiment" or "evaluation"
- `conclusion_text`: heading contains "conclusion" or "summary"
- `limitations_text`: heading contains "limitation"
- `future_work_text`: heading contains "future" or "discussion"

If multiple matching sections exist, concatenate them. If a section is missing, store empty string.

### Text cleaning
- Remove citation hash markers: `\{\{cite:[a-f0-9]{40}\}\}`
- Remove figure/table references: `\b(fig\.|figure|table)\s*\d+\b` (case-insensitive)
- Collapse multiple whitespace to single space
- Strip leading/trailing whitespace

### CSO classification
```python
from cso_classifier import CSOClassifier
classifier = CSOClassifier(modules="both", enhancement="first", delete_outliers=True)
result = classifier.classify_paper({"title": title, "abstract": abstract})
# result["syntactic"], result["semantic"], result["union"] — use "union"
# Filter to leaf nodes only (nodes with no children in CSO graph)
```

To get leaf nodes: load the CSO ontology graph. A leaf node is a concept that has no `narrower` relationships pointing away from it. The cso-classifier library bundles the ontology — ask user if they need instructions for loading it.

### Lexical filter — OPTIONAL
This step only runs if `LEXICAL_DICT_PATH` is set in `.env` and the file exists on disk.

```python
import os

def lexical_filter_enabled() -> bool:
    path = os.getenv("LEXICAL_DICT_PATH", "")
    return bool(path) and os.path.isfile(path)
```

If `lexical_filter_enabled()` returns False, skip this step entirely — all CSO tags assigned by the classifier are retained as-is. Log at INFO level: "Lexical filter disabled — using raw CSO tags."

If enabled: load `LEXICAL_DICT_PATH`. Format: list of objects `{"canonical": str, "aliases": list[str]}`. For each CSO tag assigned to a paper, check whether the paper's `title + " " + abstract + " " + intro_text` contains the canonical term or any alias (case-insensitive, whole-word match using `\b` regex). If yes, retain the tag. If no match, drop that tag for this paper.

### Cluster formation
Group papers by retained CSO leaf tags. A paper may appear in multiple clusters. Discard clusters with fewer than `CSO_MIN_CLUSTER_SIZE` papers.

If CSO tags are already included in the metadata, skip classification and use provided tags directly (still apply lexical filter if enabled).

Save all papers to `papers` table and all memberships to `cluster_membership` table. Insert a row into `cluster_progress` for each cluster with status "stage1".

### Output
Populated `papers` and `cluster_membership` tables.

---

## Stage 2 — Anchor Selection and Pairing (`stage2_anchor.py`)

### Input
All papers for a given cluster (loaded from DB).

### Embeddings
Load embeddings from `RETRIEVAL_VECTORS_PATH` (numpy array) and align rows to `paper_id` using the metadata file. Save each embedding to `papers.embedding` via `save_embedding()`.

### Anchor selection
For each paper in the cluster, compute its **claim density score**:
```python
claim_density = count_sentences(results_text) + count_sentences(conclusion_text)
# Use a simple sentence splitter: split on ". " or ".\n"
```

The paper with the highest claim density score becomes the anchor. If there is a tie, prefer the paper with the earlier `pub_date`.

### Challenger filtering
A valid challenger must:
1. Have `pub_date` strictly greater than the anchor's `pub_date`
2. Have at least one author not in the anchor's author list (to exclude self-comparisons)

If fewer than 3 valid challengers exist after filtering, log a warning and mark the cluster as "skipped" in `cluster_progress`. Do not raise an exception.

### Output
List of `(anchor_id, challenger_id)` tuples. Save to `candidate_pairs` table with status "pending". Update cluster status to "stage2".

---

## Stage 3 — Similarity Filter (`stage3_similarity.py`)

### Input
All papers in a cluster with their embeddings loaded from DB.

### Cosine similarity
```python
import numpy as np

def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T
```

### Filtering
For each candidate pair `(anchor_id, challenger_id)` from Stage 2, look up their cosine similarity from the matrix. If similarity < `SIMILARITY_THRESHOLD`, update pair status to "skipped" and do not proceed with this pair. Log how many pairs were skipped.

### Output
Remaining candidate pairs with status updated to "similarity_passed". Update cluster status to "stage3".

---

## Stage 4 — Claim Extraction (`stage4_claims.py`)

### Input
Anchor paper's `intro_text + results_text + conclusion_text` concatenated.

### vLLM client (`models/vllm_client.py`)
Implement an async HTTP client using `httpx.AsyncClient`. All model calls go through this client. Implement:

```python
async def complete(endpoint: str, model: str, prompt: str, max_tokens: int = 1000,
                   temperature: float = 0.0, logprobs: int = None) -> dict:
    # POST to endpoint with json body
    # Return full response dict
    # Raise on HTTP errors
    # Retry up to 3 times with exponential backoff on 5xx errors
```

### Phi-3 prompt (`prompts/claim_extraction.txt`)
```
You are a scientific claim extractor. Given text from a research paper, extract specific, falsifiable scientific claims.

Rules:
- Extract ONLY distinct, non-overlapping claims
- Each claim must be specific and testable, not vague
- Do not extract claims that are implied by or restate another claim in your list
- Prefer claims from Results and Conclusion over Introduction
- Maximum {MAX_CLAIMS} claims
- Return ONLY valid JSON, no preamble, no markdown fences

Output format:
{"claims": [{"text": "...", "section": "introduction|results|conclusion"}]}

Paper text:
{paper_text}
```

### Parsing
Parse the JSON response. If JSON parsing fails, retry once with the same prompt. If it fails again, log the error, save an empty claims list for this pair, and mark the pair status as "claims_failed". Do not crash.

Strip any claims shorter than 20 characters. Deduplicate by exact string match.

If the model returns more than `MAX_CLAIMS_PER_PAPER`, keep only the first `MAX_CLAIMS_PER_PAPER`.

### Output
Save claims to `claims` table. Update pair status to "claims_extracted".

---

## Stage 5 — Evidence Retrieval (`stage5_evidence.py`)

### Input
For each claim, the challenger paper's `conclusion_text + limitations_text + future_work_text` as a single string (the "evidence pool").

### Llama prompt (`prompts/evidence_retrieval.txt`)
```
You are a research assistant. Given a claim from Paper A and text from Paper B, identify the single most relevant passage from Paper B that directly addresses the claim.

Rules:
- Return the passage verbatim from the text, do not paraphrase or summarise
- The passage should be 1-4 sentences
- If no relevant passage exists, return {"evidence": null, "section": null}
- Return ONLY valid JSON

Claim from Paper A:
{claim_text}

Text from Paper B:
{evidence_pool}

Output format:
{"evidence": "...", "section": "conclusion|limitations|future_work"}
```

### Handling null evidence
If `evidence` is null, mark the claim's evidence as null. Do not pass null evidence to NLI — skip this claim entirely for NLI and mark it as "no_evidence".

### Output
Save evidence to `evidence` table. Update pair status to "evidence_retrieved".

---

## Stage 6 — Ensemble NLI (`stage6_nli.py`)

Run both models concurrently using `asyncio.gather` for each claim-evidence pair.

### Bloomz-NLI call
Use the vLLM completions endpoint. Prompt format for Bloomz (it expects a natural language NLI prompt — confirm the exact format with the user since the vLLM NLI template may differ):

```
premise: {evidence_text}
hypothesis: {claim_text}
relationship:
```

Parse the first token of the response. Map to one of: `"entailment"`, `"contradiction"`, `"neutral"`. If the response cannot be parsed, label as `"neutral"` and log a warning.

For the Bloomz score, request `logprobs=1` and extract the log probability of the first token. Convert to probability: `score = math.exp(logprob)`.

### Llama zero-shot NLI prompt (`prompts/nli_zero_shot.txt`)
```
You are a scientific reasoning assistant. Determine the logical relationship between the following hypothesis and premise from two research papers.

Premise (from Paper B):
{evidence_text}

Hypothesis (from Paper A):
{claim_text}

Classify the relationship as exactly one of:
- contradiction: the premise directly contradicts or refutes the hypothesis
- entailment: the premise supports or confirms the hypothesis
- neutral: the premise neither contradicts nor confirms the hypothesis

Rules:
- Consider scientific context carefully
- "Contradiction" requires direct logical conflict, not just different focus
- Return ONLY valid JSON

Output format:
{"label": "contradiction|entailment|neutral", "confidence": 0.0-1.0, "reasoning": "one sentence"}
```

For Llama confidence: request `logprobs=5`. Extract the log probabilities for the label token. Normalise across the three possible labels to get a calibrated probability. If logprobs are unavailable, use the `confidence` field from the JSON response as fallback.

### Ensemble logic
```python
def ensemble_label(bloomz_label, bloomz_score, llama_label, llama_confidence):
    if bloomz_label == "contradiction" and llama_label == "contradiction":
        if llama_confidence >= NLI_CONFIDENCE_THRESHOLD:
            return "contradiction"
        else:
            return "flagged"
    elif bloomz_label == "contradiction" and llama_label != "contradiction":
        return "flagged"
    elif llama_label == "contradiction" and bloomz_label != "contradiction":
        return "flagged"
    elif bloomz_label == "entailment" and llama_label == "entailment":
        return "support"
    else:
        return "neutral"
```

Save all NLI results to `nli_results` table. Update pair status to "nli_done".

---

## Stage 7 — Contradiction Typing and Explanation (`stage7_typing.py`)

Only process rows in `nli_results` where `ensemble_label = "contradiction"`.

### Taxonomy
The five types:
1. `direct_factual` — opposing factual claims about the same phenomenon
2. `methodological` — disagreement on validity or appropriateness of a method
3. `conditional` — a finding holds only under specific conditions not addressed in the other paper
4. `interpretive` — different interpretations of similar results
5. `ontological` — conflicting definitions or assumptions about a concept

### Llama typing prompt (`prompts/typing_fewshot.txt`)
Populate this file with 2 few-shot examples per type (10 examples total). The user must supply the examples from their gold standard annotations once those are ready. For now, use placeholder examples and add a TODO comment. Prompt structure:

```
You are a scientific contradiction analyst. Given a claim, evidence, and the fact that they contradict each other, classify the contradiction type and explain why.

Contradiction types:
1. direct_factual: opposing factual claims about the same phenomenon
2. methodological: disagreement on validity or appropriateness of a method
3. conditional: a finding holds only under specific conditions not addressed in the other paper
4. interpretive: different interpretations of similar results
5. ontological: conflicting definitions or assumptions about a concept

Examples:
{few_shot_examples}

Now classify this contradiction:

Claim (Paper A): {claim_text}
Evidence (Paper B): {evidence_text}

Return ONLY valid JSON:
{"type": "...", "explanation": "2-3 sentence explanation grounded in the claim and evidence"}
```

### Qwen2.5 secondary review
On a random 20% sample of contradictions, run Qwen2.5 (ask user for its vLLM endpoint). Prompt it with the same claim + evidence and ask it to independently classify the contradiction type. Save its output to `contradictions.reviewer_label` and `contradictions.reviewer_notes`. Use this for agreement analysis but do not override the primary Llama label.

### Output
Save to `contradictions` table. Update pair status to "typed". Update cluster status to "done".

---

## Celery Workers (`workers/`)

### `celery_app.py`
```python
from celery import Celery
app = Celery("contradiction_pipeline", broker=REDIS_URL, backend=REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.task_acks_late = True          # re-queue on worker crash
app.conf.task_reject_on_worker_lost = True
app.conf.worker_prefetch_multiplier = 1
```

### `workers/tasks.py`
Define one Celery task per cluster that runs stages 2–7 sequentially:

```python
@app.task(bind=True, max_retries=CELERY_TASK_RETRIES, default_retry_delay=CELERY_RETRY_BACKOFF)
def process_cluster(self, cluster_id: str):
    try:
        run_stage2(cluster_id)
        run_stage3(cluster_id)
        run_stage4(cluster_id)
        run_stage5(cluster_id)
        run_stage6(cluster_id)
        run_stage7(cluster_id)
        set_cluster_status(cluster_id, "done")
    except Exception as exc:
        set_cluster_status(cluster_id, "error", error_message=str(exc))
        raise self.retry(exc=exc)
```

Stage 1 runs separately as a blocking pre-processing step before workers are launched, since it must populate the DB before tasks can be dispatched.

### Resumption logic
At startup of `run_pipeline.py`, query `cluster_progress` for all clusters with status != "done". For each, dispatch a `process_cluster` task only if one is not already queued (check Redis for active tasks). This ensures safe restart after crash.

---

## Main Entry Point (`scripts/run_pipeline.py`)

```python
def main():
    init_db(SQLITE_DB_PATH)
    
    # Stage 1: blocking, run once
    if no clusters exist in DB:
        run_stage1()
    
    # Dispatch Celery tasks for all pending clusters
    clusters = get_pending_clusters()
    for cluster_id in clusters:
        process_cluster.delay(cluster_id)
    
    print(f"Dispatched {len(clusters)} cluster tasks")
```

---

## Logging

Use Python's `logging` module with a `RotatingFileHandler` (max 50MB per file, 5 backups). Log to both file and stdout.

Log the following at INFO level:
- Cluster start/end
- Number of pairs generated per cluster
- Number of claims extracted per pair
- NLI label counts per cluster

Log the following at WARNING level:
- Clusters skipped due to insufficient challengers
- Claim extraction JSON parse failures
- NLI response parse failures
- Pairs with null evidence

Log the following at ERROR level:
- Any unhandled exception with full traceback

---

## Evaluation (`evaluation/`) — OPTIONAL

This entire section is optional. The pipeline runs and produces results in SQLite with or without it. Only implement evaluation if `GOLD_STANDARD_PATH` is set in `.env` and the file actually exists on disk.

### Guard pattern — apply this everywhere in evaluation code

```python
import os

def evaluation_available() -> bool:
    path = os.getenv("GOLD_STANDARD_PATH", "")
    return bool(path) and os.path.isfile(path)
```

Call `evaluation_available()` at the top of `run_evaluation.py`. If it returns False, print a clear message and exit gracefully:

```python
if not evaluation_available():
    print("No gold standard found. Set GOLD_STANDARD_PATH in .env to enable evaluation.")
    print("Pipeline results are available in the SQLite database at:", SQLITE_DB_PATH)
    sys.exit(0)
```

### `gold_standard.py`
Load a gold standard CSV with columns: `anchor_id, challenger_id, label, contradiction_type`.
- `label`: one of `contradiction`, `support`, `neutral`, `unrelated`
- `contradiction_type`: one of the 5 taxonomy types, or empty string if label is not contradiction

Match rows to internal `candidate_pairs` by looking up `(anchor_id, challenger_id)` as arXiv IDs. Log a warning for any gold standard row that cannot be matched to a pair in the DB — do not crash.

### `metrics.py`
Implement the following functions. All are only called when evaluation is available:

- `precision_recall_f1(y_true, y_pred, positive_class="contradiction")` — returns dict with precision, recall, f1
- `macro_f1(y_true, y_pred)` — across all 4 label classes
- `bootstrap_ci(y_true, y_pred, n_iterations=1000, metric_fn=None)` — returns (mean, lower_95, upper_95). Use numpy random sampling with replacement.
- `mcnemar_test(y_true, pred_a, pred_b)` — returns p-value. Use scipy.stats.

### `run_evaluation.py`
Only runs if `evaluation_available()` is True. Produces a printed report covering:
- Binary contradiction detection: precision, recall, F1 with 95% bootstrap CI
- Macro F1 across all 4 classes
- Confusion matrix counts
- Breakdown of F1 per contradiction type
- McNemar test comparing full pipeline vs Bloomz-only vs Llama-only (pull those labels from `nli_results` table)

Do not produce charts or files — print to stdout only unless the user asks for file output.

---

## Prompts — Important Notes for Codex

All prompt templates live in `/prompts/*.txt`. Load them at runtime using `open()`, do not hardcode in Python files. Template variables use `{variable_name}` format. Use Python's `str.format()` or a simple regex substitution to fill them.

---

## Error Handling Rules

1. Never let a single pair failure crash a cluster task — catch exceptions at the pair level, log them, and continue.
2. Never let a single cluster failure crash the whole pipeline — Celery retries handle this.
3. If a model endpoint is unreachable for more than 3 retries, mark all pairs in the current cluster as "error" and re-raise so Celery reschedules.
4. Always validate JSON responses before accessing fields — use `try/except json.JSONDecodeError`.
5. If an embedding is missing from the DB for a paper, skip that paper and log a warning.

---

## Environment Variables (`.env` file)

```
RETRIEVAL_VECTORS_PATH=/path/to/retrieval_vectors.npy
METADATA_PATH=/path/to/metadata.jsonl
SQLITE_DB_PATH=pipeline_state.db
REDIS_URL=redis://localhost:6379/0
PHI3_ENDPOINT=http://localhost:8001/v1/completions
BLOOMZ_ENDPOINT=http://localhost:8002/v1/completions
LLAMA_ENDPOINT=http://localhost:8003/v1/completions
QWEN_ENDPOINT=http://localhost:8004/v1/completions
LOG_PATH=logs/pipeline.log

# Optional — leave empty to disable lexical filtering
LEXICAL_DICT_PATH=

# Optional — leave empty to skip evaluation entirely
GOLD_STANDARD_PATH=
```

---

## HPC Deployment — Singularity and SLURM

This section covers everything needed to run the pipeline on an HPC cluster. All commands below assume SLURM as the scheduler and Singularity as the container runtime. The user will need to confirm their HPC's module system and scratch path — ask them before writing these scripts.

### Step 1 — Build the Singularity container

The `Dockerfile` defines the full environment. Build it locally and push to Docker Hub, then pull it on the HPC and convert to a `.sif` file. Or build directly on the HPC if Docker is available there (ask user).

`hpc/container/Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Python deps — install everything here so the container is self-contained
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into the image
COPY . .
```

`hpc/container/build_container.sh` — run this on the HPC login node:
```bash
#!/bin/bash
# Run once to build the Singularity image from Docker Hub
# Replace YOUR_DOCKERHUB_USER with the actual Docker Hub username

mkdir -p containers

# Option A: pull from Docker Hub (preferred)
singularity build ./containers/contradiction_pipeline.sif \
    docker://YOUR_DOCKERHUB_USER/contradiction_pipeline:latest

# Option B: build directly from local Dockerfile if Docker is available on HPC
# singularity build --fakeroot ./containers/contradiction_pipeline.sif \
#     docker-daemon://contradiction_pipeline:latest
```

Ask the user: do they have Docker Hub access to push the image, or should Option B (local build) be used?

### Step 2 — HPC environment file

`hpc/env/hpc.env` — user must fill in their actual paths:
```
# HPC paths — use $SCRATCH or $WORK for large files
RETRIEVAL_VECTORS_PATH=$SCRATCH/path/to/retrieval_vectors.npy
METADATA_PATH=$SCRATCH/path/to/metadata.jsonl
SQLITE_DB_PATH=$SCRATCH/contradiction_pipeline/pipeline_state.db
LOG_PATH=$SCRATCH/contradiction_pipeline/logs/pipeline.log

# On HPC, Redis is not available. Set USE_CELERY=false to use sequential mode.
USE_CELERY=false

# vLLM model servers run as separate SLURM jobs on the same node.
# These endpoints use localhost because all jobs in a step share the node.
PHI3_ENDPOINT=http://localhost:8001/v1/completions
BLOOMZ_ENDPOINT=http://localhost:8002/v1/completions
LLAMA_ENDPOINT=http://localhost:8003/v1/completions
QWEN_ENDPOINT=http://localhost:8004/v1/completions

PHI3_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
BLOOMZ_MODEL_NAME=bigscience/bloomz-3b
LLAMA_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct

SPECTER_MODEL=allenai/specter2_base
SPECTER_ADAPTER=allenai/specter2
SPECTER_BATCH_SIZE=64
EMBEDDING_DIM=768

SIMILARITY_THRESHOLD=0.75
NLI_CONFIDENCE_THRESHOLD=0.65
ENSEMBLE_AGREE_ONLY=true
MAX_CLAIMS_PER_PAPER=15
CSO_MIN_CLUSTER_SIZE=20

LEXICAL_DICT_PATH=
GOLD_STANDARD_PATH=
```

### Step 3 — Sequential mode (no Redis/Celery on HPC)

In `scripts/run_pipeline.py`, check the `USE_CELERY` env var:

```python
USE_CELERY = os.getenv("USE_CELERY", "true").lower() == "true"

def main():
    init_db(SQLITE_DB_PATH)
    if not clusters_exist_in_db():
        run_stage1()

    clusters = get_pending_clusters()

    if USE_CELERY:
        # Local mode: dispatch to Redis queue
        for cluster_id in clusters:
            process_cluster.delay(cluster_id)
        print(f"Dispatched {len(clusters)} cluster tasks to Celery")
    else:
        # HPC mode: run clusters sequentially in this process
        # SLURM handles parallelism by submitting multiple jobs with different CLUSTER_BATCH_START
        batch_start = int(os.getenv("CLUSTER_BATCH_START", 0))
        batch_size  = int(os.getenv("CLUSTER_BATCH_SIZE", len(clusters)))
        batch = clusters[batch_start : batch_start + batch_size]
        print(f"HPC sequential mode: processing {len(batch)} clusters (batch {batch_start})")
        for cluster_id in batch:
            try:
                process_cluster_sequential(cluster_id)
            except Exception as e:
                logging.error(f"Cluster {cluster_id} failed: {e}", exc_info=True)
                set_cluster_status(cluster_id, "error", error_message=str(e))
```

`process_cluster_sequential(cluster_id)` is the same logic as the Celery task but called directly — no queue, no broker. Implement it in `workers/tasks.py` alongside the Celery version so both share code.

### Step 4 — SLURM job scripts

All job scripts in `hpc/jobs/` follow the same pattern: load the Singularity container, pass the `.env` file, and run the relevant Python script. The user must fill in `--partition`, `--time`, and `--gres` values for their specific cluster — add TODO comments for these.

---

`hpc/jobs/01_stage1.sh` — ingest UnarXive and build clusters (CPU-only, high RAM):
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=TODO_ASK_USER      # Ask user for their CPU partition name
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage1-%J.out
#SBATCH --error=logs/stage1-%J.err
#SBATCH --job-name="contradiction-stage1"

mkdir -p logs

srun singularity exec --bind $SCRATCH:$SCRATCH \
    ./containers/contradiction_pipeline.sif \
    python scripts/run_pipeline.py \
        --env hpc/env/hpc.env \
        --stages 1
```

---

`hpc/jobs/02_embed.sh` — SPECTER 2 embeddings for all clusters (GPU):
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=TODO_ASK_USER      # Ask user for their GPU partition name
#SBATCH --time=04:00:00
#SBATCH --output=logs/embed-%J.out
#SBATCH --error=logs/embed-%J.err
#SBATCH --job-name="contradiction-embed"

mkdir -p logs

srun singularity exec --nv --bind $SCRATCH:$SCRATCH \
    ./containers/contradiction_pipeline.sif \
    python scripts/run_pipeline.py \
        --env hpc/env/hpc.env \
        --stages 2,3
```

Note: `--nv` flag passes GPU access through to the container. Required for CUDA.

---

`hpc/jobs/03_models.sh` — launch all vLLM model servers (GPU, runs in background):
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4              # 4 GPUs: one per model server + headroom
#SBATCH --partition=TODO_ASK_USER
#SBATCH --time=24:00:00           # Long-running: stays alive while pipeline jobs run
#SBATCH --output=logs/models-%J.out
#SBATCH --error=logs/models-%J.err
#SBATCH --job-name="contradiction-models"

mkdir -p logs

SIF=./containers/contradiction_pipeline.sif
ENV=hpc/env/hpc.env

# Launch each vLLM server on a separate port, backgrounded
singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python -m vllm.entrypoints.openai.api_server \
        --model microsoft/Phi-3-mini-4k-instruct \
        --port 8001 --gpu-memory-utilization 0.3 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python -m vllm.entrypoints.openai.api_server \
        --model bigscience/bloomz-3b \
        --port 8002 --gpu-memory-utilization 0.25 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --port 8003 --gpu-memory-utilization 0.35 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8004 --gpu-memory-utilization 0.3 &

# Wait for all servers to be ready before this job script exits
echo "Waiting for model servers to be ready..."
sleep 60
echo "Model servers running. Job will stay alive for wall time."
wait
```

Ask the user: what are the actual model names/paths on their HPC? Models may need to be pre-downloaded to `$SCRATCH` if the compute nodes have no internet access. If so, add a download step.

---

`hpc/jobs/04_pipeline.sh` — run stages 4-7 over all clusters (GPU, can run in parallel batches):
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=TODO_ASK_USER
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline-%J.out
#SBATCH --error=logs/pipeline-%J.err
#SBATCH --job-name="contradiction-pipeline"

# CLUSTER_BATCH_START and CLUSTER_BATCH_SIZE can be passed via --export
# to run different batches of clusters in parallel
# Example: sbatch --export=CLUSTER_BATCH_START=0,CLUSTER_BATCH_SIZE=50 hpc/jobs/04_pipeline.sh
#          sbatch --export=CLUSTER_BATCH_START=50,CLUSTER_BATCH_SIZE=50 hpc/jobs/04_pipeline.sh

mkdir -p logs

srun singularity exec --nv --bind $SCRATCH:$SCRATCH \
    ./containers/contradiction_pipeline.sif \
    python scripts/run_pipeline.py \
        --env hpc/env/hpc.env \
        --stages 4,5,6,7
```

### Step 5 — Submitting jobs in order

Create `hpc/submit_all.sh` as a convenience script that submits all jobs in dependency order using SLURM `--dependency`:

```bash
#!/bin/bash
# Submit all pipeline jobs in sequence.
# Each job only starts after the previous one completes successfully.

JOB1=$(sbatch --parsable hpc/jobs/01_stage1.sh)
echo "Submitted stage1 job: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 hpc/jobs/02_embed.sh)
echo "Submitted embed job: $JOB2"

JOB3=$(sbatch --parsable hpc/jobs/03_models.sh)
echo "Submitted model servers job: $JOB3 (running independently)"

# Pipeline job depends on both embedding AND model servers being ready
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 hpc/jobs/04_pipeline.sh)
echo "Submitted pipeline job: $JOB4"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs in: logs/"
```

### Step 6 — Local reproducibility (non-HPC)

Anyone cloning the repo on a local machine runs:

```bash
git clone https://github.com/YOUR_USER/contradiction_pipeline.git
cd contradiction_pipeline

# Copy and fill in the local env file
cp hpc/env/local.env .env
nano .env   # set UNARXIVE_PATH, model endpoints, etc.

# Option A: Docker
docker build -t contradiction_pipeline .
docker run --gpus all -v $(pwd):/app contradiction_pipeline \
    python scripts/run_pipeline.py --env .env

# Option B: plain virtualenv (no container)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --env .env
```

`hpc/env/local.env` example:
```
RETRIEVAL_VECTORS_PATH=./data/processed/retrieval_vectors.npy
METADATA_PATH=./data/processed/metadata.jsonl
SQLITE_DB_PATH=./pipeline_state.db
LOG_PATH=./logs/pipeline.log
USE_CELERY=true
REDIS_URL=redis://localhost:6379/0
PHI3_ENDPOINT=http://localhost:8001/v1/completions
BLOOMZ_ENDPOINT=http://localhost:8002/v1/completions
LLAMA_ENDPOINT=http://localhost:8003/v1/completions
QWEN_ENDPOINT=http://localhost:8004/v1/completions
PHI3_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
BLOOMZ_MODEL_NAME=bigscience/bloomz-3b
LLAMA_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
SPECTER_MODEL=allenai/specter2_base
SPECTER_ADAPTER=allenai/specter2
SPECTER_BATCH_SIZE=32
EMBEDDING_DIM=768
SIMILARITY_THRESHOLD=0.75
NLI_CONFIDENCE_THRESHOLD=0.65
ENSEMBLE_AGREE_ONLY=true
MAX_CLAIMS_PER_PAPER=15
CSO_MIN_CLUSTER_SIZE=20
LEXICAL_DICT_PATH=
GOLD_STANDARD_PATH=
```

### Binding scratch storage

All SLURM scripts use `--bind $SCRATCH:$SCRATCH` to make the HPC scratch filesystem visible inside the Singularity container. The `SQLITE_DB_PATH` and `LOG_PATH` in `hpc.env` must point to locations under `$SCRATCH` or `$WORK` — not the repo directory, which is read-only inside the container. If the user's HPC uses a different environment variable for scratch storage (e.g. `$TMPDIR`, `$WORK`, `/scratch/$USER`), ask them and update the bind path accordingly.

---

## Open Questions for Codex to Ask the User

Before writing any code, ask the user to confirm or provide:

1. **Metadata file** — What is the exact path and format (CSV, JSONL, Parquet) that maps each row in `retrieval_vectors.npy` to `paper_id`, title, abstract, authors, and sectioned text?
2. **HPC scratch variable** — Is scratch storage accessed via `$SCRATCH`, `$WORK`, `/scratch/$USER`, or something else? This determines the `--bind` path in all SLURM scripts.
3. **HPC partition names** — What are the names of the CPU and GPU partitions on this cluster? Replaces the `TODO_ASK_USER` placeholders in all SLURM scripts.
4. **Internet access on compute nodes** — Can compute nodes download HuggingFace models at runtime, or must models be pre-downloaded to scratch before job submission? If pre-download is needed, a separate download script must be added.
5. **Docker Hub access** — Can the user push a Docker image to Docker Hub for Singularity to pull? Or should the container be built directly on the HPC login node using `--fakeroot`?
6. **Lexical dictionary** — Do you have a curated AI terms JSON file? If not, leave `LEXICAL_DICT_PATH` empty — the filter skips automatically.
7. **vLLM ports** — Confirm the ports for all four model endpoints. Default is 8001-8004 — confirm there are no port conflicts on the cluster.
8. **Bloomz NLI prompt format** — The exact input format expected by the Bloomz vLLM endpoint. NLI models sometimes require specific prompt templates.
9. **Gold standard CSV** — Do you have one? If not, leave `GOLD_STANDARD_PATH` empty — evaluation is skipped entirely.
10. **Few-shot typing examples** — Do you have 2 examples per contradiction type? If not, placeholders with TODO comments will be used.
11. **Cluster subset for first test run** — Should the first run cover a small subset (e.g. 5 clusters) or the full corpus?

---

---

## HTML Report Generator (`scripts/generate_report.py`)

After the pipeline finishes, run this script to produce a single self-contained `report.html` file. No server, no ports, no dependencies at runtime — just copy the file to your local machine with `scp` and open it in a browser.

```bash
# On HPC after pipeline completes
python scripts/generate_report.py --env hpc/env/hpc.env --output $SCRATCH/contradiction_pipeline/report.html

# Copy to local machine
scp your_user@hpc.university.edu:$SCRATCH/contradiction_pipeline/report.html ./report.html
```

### What the report must contain

The script reads entirely from SQLite — no model calls, no network. It generates one HTML string and writes it to the output path.

**Section 1 — Pipeline summary**
A top-level stats table showing:
- Total papers ingested
- Total clusters formed
- Total candidate pairs evaluated
- Total contradictions found
- Total pairs flagged (Bloomz and Llama disagreed)
- Total pairs skipped (no evidence found)
- Clusters with status "error" (count + list of cluster IDs)
- Pipeline run date (taken from the latest `updated_at` in `cluster_progress`)

**Section 2 — Contradiction type breakdown**
A horizontal bar chart showing counts per contradiction type (direct_factual, methodological, conditional, interpretive, ontological). Use Chart.js loaded from `https://cdn.jsdelivr.net/npm/chart.js`. The chart must render in the HTML file with no server — use inline `<canvas>` and a `<script>` block.

**Section 3 — Per-cluster summary table**
A sortable HTML table (sort by clicking column headers, implemented in vanilla JS — no library needed) with columns:
- Cluster ID (CSO term)
- Paper count
- Pair count
- Contradiction count
- Flagged count
- Status (colour-coded: green = done, red = error, yellow = pending)

**Section 4 — Contradiction browser**
A paginated list of contradiction cards (20 per page, Previous/Next buttons in vanilla JS). Each card shows:
- Anchor paper title and arXiv ID (link to `https://arxiv.org/abs/{paper_id}`)
- Challenger paper title and arXiv ID (link)
- Cluster ID
- Contradiction type (colour-coded badge)
- Claim text (from `claims` table)
- Evidence text (from `evidence` table)
- Llama explanation (from `contradictions` table)
- Ensemble confidence (Llama confidence score, formatted as a percentage)
- Bloomz label and Llama label side by side so disagreements are visible

**Section 5 — Flagged pairs browser**
Same card layout as Section 4 but for pairs where `ensemble_label = "flagged"` (Bloomz and Llama disagreed). These are candidates for manual annotation. Each card has a visible "Bloomz said X / Llama said Y" label so the user can see the disagreement at a glance.

**Section 6 — Sample neutral/support pairs**
10 randomly sampled pairs per label class (neutral, support) shown in a collapsed `<details>` block. Useful for sanity-checking that the pipeline is not over-predicting contradictions.

### Implementation rules for Codex

- The entire report is one HTML file. All CSS is inline in a `<style>` block. All JS is inline in `<script>` blocks. No external files except Chart.js from the CDN.
- Use a clean, readable sans-serif font. Black text on white background. Contradiction type badges use distinct background colours — one per type, defined as a JS object mapping type name to hex colour.
- All data is embedded as a JSON object in a `<script>` block at the top of the body: `const REPORT_DATA = {...};`. The JS reads from this object — it never fetches anything at runtime.
- The Python script builds this JSON object from SQLite queries, then uses Python's `str` formatting or a Jinja2 template to inject it into the HTML. Ask user if Jinja2 is acceptable or if plain string formatting is preferred.
- Paginator state is managed in JS variables — no page reload on navigation.
- The sortable table must sort correctly for both string and numeric columns. Clicking a column header twice reverses sort order.
- arXiv links must open in a new tab (`target="_blank"`).
- The report must be readable on a laptop screen without horizontal scrolling. Max content width 1100px, centred.
- Add a prominent note at the top of the report: "Generated from pipeline_state.db on {date}. Flagged pairs have not been human-reviewed."

### SLURM job for report generation

Add `hpc/jobs/05_report.sh`:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=TODO_ASK_USER     # CPU partition, no GPU needed
#SBATCH --time=00:30:00
#SBATCH --output=logs/report-%J.out
#SBATCH --error=logs/report-%J.err
#SBATCH --job-name="contradiction-report"

mkdir -p logs

srun singularity exec --bind $SCRATCH:$SCRATCH \
    ./containers/contradiction_pipeline.sif \
    python scripts/generate_report.py \
        --env hpc/env/hpc.env \
        --output $SCRATCH/contradiction_pipeline/report.html

echo "Report written to $SCRATCH/contradiction_pipeline/report.html"
echo "Copy to local machine with:"
echo "  scp $USER@$(hostname -f):$SCRATCH/contradiction_pipeline/report.html ./report.html"
```

Add this job to `hpc/submit_all.sh` as a final step with `--dependency=afterok:$JOB4`.



In `tests/test_each_stage.py`, write unit tests for:
- Section extraction from a sample `body_text` list
- Text cleaning regex
- Lexical filter matching (canonical + alias)
- Cosine similarity matrix correctness
- Claim JSON parsing (valid case + malformed case)
- Evidence null handling
- Ensemble label logic (all 4 cases)
- DB save/load round-trip for embeddings

Use `pytest`. Mock all vLLM HTTP calls using `pytest-httpx` or `unittest.mock`.

---

## Deliverables Checklist

**Core pipeline (always required):**
- [ ] All 7 stage modules implemented
- [ ] SQLite schema initialised and all helpers working
- [ ] Pipeline runs end-to-end on a single test cluster
- [ ] `USE_CELERY=true` mode (local) and `USE_CELERY=false` mode (HPC sequential) both working
- [ ] Prompts loaded from files, not hardcoded
- [ ] All paths come from `.env`, zero hardcoded paths in source
- [ ] `--env` CLI argument working on all entry point scripts
- [ ] Logging working to file and stdout
- [ ] `scripts/generate_report.py` produces a valid self-contained `report.html`
- [ ] Report contains all 6 sections: summary, type chart, cluster table, contradiction browser, flagged browser, neutral/support samples
- [ ] Unit tests passing
- [ ] README with both local and HPC setup instructions, including scp command for copying report

**HPC (always required):**
- [ ] `Dockerfile` builds successfully
- [ ] `hpc/container/build_container.sh` produces a working `.sif`
- [ ] All 4 SLURM job scripts in `hpc/jobs/` with correct `--bind` paths
- [ ] `hpc/submit_all.sh` chains jobs with `--dependency=afterok`
- [ ] `hpc/env/hpc.env` and `hpc/env/local.env` both present with comments
- [ ] Model servers launch correctly inside Singularity with `--nv` flag

**Optional — only if user supplies files:**
- [ ] Lexical filter active (requires `LEXICAL_DICT_PATH` to point to a valid JSON file)
- [ ] Evaluation metrics and report (requires `GOLD_STANDARD_PATH` to point to a valid CSV)
- [ ] Few-shot typing examples populated (requires user to supply examples per contradiction type)