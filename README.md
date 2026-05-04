# Contradiction Pipeline

This pipeline detects and categorizes contradictions between papers using precomputed retrieval embeddings and a metadata mapping file.

## Requirements
- Python 3.11
- vLLM endpoints for Phi-3, Bloomz, Llama, and Qwen (for optional review)
- Retrieval vectors and metadata mapping file

## Quick start (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp hpc/env/local.env .env
# Update RETRIEVAL_VECTORS_PATH and METADATA_PATH in .env
python scripts/run_pipeline.py --env .env
```

This setup is fully isolated to `contradictions/.venv` and does not require importing or installing from other project folders.

Optional if you want to host model servers yourself in this venv:
```bash
pip install -r requirements.server.txt
```

## Mandatory CSO Classifier Setup (contradictions-only venv)
If you must run Stage 1 with CSO classifier (not category fallback), use:

```bash
bash scripts/install_cso_stack.sh
```

This installs the compatible CSO stack into `contradictions/.venv`, including:
- `cso-classifier` (installed with `--no-deps`)
- compatible runtime libs from `requirements.cso.txt`
- spaCy model `en_core_web_sm`
- NLTK `stopwords`

Set this in `.env` for the validated configuration:
```
CSO_DELETE_OUTLIERS=false
```

## HPC
- Update hpc/env/hpc.env with scratch paths and model endpoints.
- Fill in TODO partitions in hpc/jobs/*.sh.
- Submit pipeline: `bash hpc/submit_all.sh`.

## Metadata format
For your processed dataset, use:
- `METADATA_PATH=.../retrieval_index_meta.parquet`
- `CHUNKS_METADATA_PATH=.../retrieval_chunks.parquet`

Detected field names:
- `retrieval_index_meta.parquet`: `vector_row`, `chunk_id`, `source_type`, `doc_id`, `title`, `date`, `categories`, `representation_type`, `section_label`, `chunk_index`, `citation_id`, `row_group`, `row_in_group`
- `retrieval_chunks.parquet`: `chunk_id`, `source_type`, `doc_id`, `title`, `date`, `categories`, `representation_type`, `section_label`, `chunk_index`, `citation_id`, `text`

The loader joins both on `chunk_id`, orders by `vector_row`, maps `doc_id -> paper_id`, and splits `text` by `[SEP]` to derive abstract when available.
Only rows where `source_type == "paper"` are used by the pipeline. Non-paper sources (for example patents) are ignored.

Also supported for custom metadata: JSON, JSONL, CSV, TSV, Parquet.

## Reports
After completion:
```bash
python scripts/generate_report.py --env .env --output ./report.html
```
