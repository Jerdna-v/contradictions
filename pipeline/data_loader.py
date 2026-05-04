import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pipeline.text_utils import extract_sections

_EMBEDDING_CACHE = None


def _paper_limit() -> int:
    value = os.getenv("RETRIEVAL_MAX_PAPERS", "100000").strip()
    try:
        limit = int(value)
    except ValueError:
        limit = 100000
    return max(1, limit)


def _is_paper(record: Dict[str, object]) -> bool:
    return str(record.get("source_type") or "").strip().lower() == "paper"


def _load_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    raise ValueError("Unsupported JSON structure for metadata")


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_csv(path: str, delimiter: str = ",") -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [row for row in reader]


def load_metadata(path: str) -> List[Dict[str, object]]:
    if not path:
        raise ValueError("METADATA_PATH is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    lower = path.lower()
    if lower.endswith(".jsonl"):
        return _load_jsonl(path)
    if lower.endswith(".json"):
        return _load_json(path)
    if lower.endswith(".csv"):
        return _load_csv(path, ",")
    if lower.endswith(".tsv"):
        return _load_csv(path, "\t")
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    raise ValueError("Unsupported metadata format: " + path)


def normalize_record(record: Dict[str, object]) -> Dict[str, object]:
    # UL-FRI processed format: title and abstract are packed in `text` as
    # "<title> [SEP] <abstract>". We derive missing fields with safe defaults.
    if "doc_id" in record and "text" in record:
        packed = record.get("text") or ""
        if "[SEP]" in packed:
            left, right = packed.split("[SEP]", 1)
            title = (record.get("title") or left).strip()
            abstract = right.strip()
        else:
            title = (record.get("title") or "").strip()
            abstract = packed.strip()

        return {
            "paper_id": record.get("doc_id"),
            "source_type": record.get("source_type", "paper"),
            "title": title,
            "abstract": abstract,
            "pub_date": record.get("date") or "",
            "authors": [],
            "body_text": [],
            "cso_tags": [],
            "categories": record.get("categories") or "",
            "intro_text": "",
            "results_text": "",
            "conclusion_text": "",
            "limitations_text": "",
            "future_work_text": "",
        }

    body_text = record.get("body_text") or []
    sections = extract_sections(body_text) if body_text else {
        "intro_text": "",
        "results_text": "",
        "conclusion_text": "",
        "limitations_text": "",
        "future_work_text": "",
    }

    return {
        "paper_id": record.get("paper_id") or record.get("id"),
        "source_type": record.get("source_type", "paper"),
        "title": record.get("title") or "",
        "abstract": record.get("abstract") or "",
        "pub_date": record.get("pub_date") or record.get("date") or "",
        "authors": record.get("authors") or [],
        "body_text": body_text,
        "cso_tags": record.get("cso_tags"),
        "categories": record.get("categories") or "",
        **sections,
    }


def load_embeddings_map(metadata_path: str, vectors_path: str) -> Dict[str, np.ndarray]:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE

    if not vectors_path:
        raise ValueError("RETRIEVAL_VECTORS_PATH is required")
    if not os.path.isfile(vectors_path):
        raise FileNotFoundError(vectors_path)

    chunks_path = os.getenv("CHUNKS_METADATA_PATH", "")
    if metadata_path.endswith("retrieval_index_meta.parquet") and chunks_path:
        records = load_ul_fri_processed_metadata(metadata_path, chunks_path)
    else:
        records = [normalize_record(r) for r in load_metadata(metadata_path)]
    vectors = np.load(vectors_path)

    # If the vectors file contains more rows than the selected metadata rows,
    # slice the vectors to match the records. If it contains fewer rows than
    # records, error out since alignment cannot be guaranteed.
    if len(vectors) < len(records):
        raise ValueError("Embeddings rows fewer than selected metadata rows")
    if len(vectors) > len(records):
        vectors = vectors[: len(records)]

    mapping = {}
    for record, vector in zip(records, vectors):
        if (record.get("source_type") or "paper") != "paper":
            continue
        paper_id = record.get("paper_id")
        if not paper_id:
            continue
        mapping[paper_id] = np.asarray(vector, dtype=np.float32)

    _EMBEDDING_CACHE = mapping
    return mapping


def load_ul_fri_processed_metadata(index_meta_path: str, chunks_path: str) -> List[Dict[str, object]]:
    index_df = pd.read_parquet(index_meta_path)
    chunks_df = pd.read_parquet(chunks_path)

    merged = index_df.merge(
        chunks_df[["chunk_id", "text"]],
        on="chunk_id",
        how="left",
    )

    # Use vector row order to align metadata with retrieval_vectors.npy rows.
    merged = merged.sort_values("vector_row")

    normalized = [normalize_record(r) for r in merged.to_dict(orient="records")]
    papers_only = [r for r in normalized if _is_paper(r)]
    return papers_only[:_paper_limit()]


def load_papers_with_metadata(metadata_path: str) -> List[Dict[str, object]]:
    chunks_path = os.getenv("CHUNKS_METADATA_PATH", "")
    if metadata_path.endswith("retrieval_index_meta.parquet") and chunks_path:
        return load_ul_fri_processed_metadata(metadata_path, chunks_path)

    records = load_metadata(metadata_path)
    normalized = [normalize_record(r) for r in records]
    # Return only papers; embeddings mapping will align rows and skip non-papers
    papers_only = [r for r in normalized if _is_paper(r)]
    return papers_only[:_paper_limit()]
