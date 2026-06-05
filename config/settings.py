import os

# All values here are defaults. Override via .env file.

RETRIEVAL_VECTORS_PATH = os.getenv("RETRIEVAL_VECTORS_PATH", "")
METADATA_PATH = os.getenv("METADATA_PATH", "")
CHUNKS_METADATA_PATH = os.getenv("CHUNKS_METADATA_PATH", "")
RAW_PAPERS_PATH = os.getenv("RAW_PAPERS_PATH", "")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "pipeline_state.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_PATH = os.getenv("LOG_PATH", "logs/pipeline.log")

# CSO
CSO_MIN_CLUSTER_SIZE = int(os.getenv("CSO_MIN_CLUSTER_SIZE", "20"))
LEXICAL_DICT_PATH = os.getenv("LEXICAL_DICT_PATH", "")
CSO_DELETE_OUTLIERS = os.getenv("CSO_DELETE_OUTLIERS", "false").lower() == "true"
CSO_BATCH_WORKERS = int(os.getenv("CSO_BATCH_WORKERS", "1"))

# Embeddings
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Similarity
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# vLLM endpoints
PHI3_ENDPOINT = os.getenv("PHI3_ENDPOINT", "http://localhost:8001/v1/completions")
BLOOMZ_ENDPOINT = os.getenv("BLOOMZ_ENDPOINT", "http://localhost:8002/v1/completions")
LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT", "http://localhost:8003/v1/completions")
QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "http://localhost:8004/v1/completions")

PHI3_MODEL_NAME = os.getenv("PHI3_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
BLOOMZ_MODEL_NAME = os.getenv("BLOOMZ_MODEL_NAME", "bigscience/bloomz-3b")
LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# NLI
NLI_CONFIDENCE_THRESHOLD = float(os.getenv("NLI_CONFIDENCE_THRESHOLD", "0.65"))
ENSEMBLE_AGREE_ONLY = os.getenv("ENSEMBLE_AGREE_ONLY", "true").lower() == "true"

# Celery
CELERY_TASK_RETRIES = int(os.getenv("CELERY_TASK_RETRIES", "3"))
CELERY_RETRY_BACKOFF = int(os.getenv("CELERY_RETRY_BACKOFF", "60"))

# Claim extraction
MAX_CLAIMS_PER_PAPER = int(os.getenv("MAX_CLAIMS_PER_PAPER", "15"))

# Retrieval max rows (optional). Empty means no limit; normally we do NOT
# truncate metadata rows because embeddings must align with the full index.
RETRIEVAL_MAX_ROWS = os.getenv("RETRIEVAL_MAX_ROWS", "")
if RETRIEVAL_MAX_ROWS:
	try:
		RETRIEVAL_MAX_ROWS = int(RETRIEVAL_MAX_ROWS)
	except Exception:
		RETRIEVAL_MAX_ROWS = None
# Optional evaluation
GOLD_STANDARD_PATH = os.getenv("GOLD_STANDARD_PATH", "")

# Mode
USE_CELERY = os.getenv("USE_CELERY", "true").lower() == "true"
