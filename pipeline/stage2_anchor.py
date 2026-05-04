import datetime as dt
import json
import logging
import uuid
from typing import Dict, List, Optional

from db.schema import (
    bulk_save_embeddings,
    bulk_save_pairs,
    get_papers_for_cluster,
    set_cluster_status,
)
from pipeline.data_loader import load_embeddings_map
from pipeline.text_utils import count_sentences

LOGGER = logging.getLogger("contradiction_pipeline")


def _parse_date(value: str) -> Optional[dt.date]:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _parse_json_list(value: object) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _claim_density(paper: Dict[str, object]) -> int:
    return count_sentences(paper.get("results_text", "")) + count_sentences(
        paper.get("conclusion_text", "")
    )


def run_stage2(db_path: str, cluster_id: str, metadata_path: str, vectors_path: str, embeddings_map: Optional[Dict] = None) -> None:
    papers = get_papers_for_cluster(db_path, cluster_id)
    if not papers:
        LOGGER.warning("No papers found for cluster %s", cluster_id)
        return

    if embeddings_map is None:
        embeddings_map = load_embeddings_map(metadata_path, vectors_path)

    embedding_rows = []
    for paper in papers:
        paper_id = paper.get("paper_id")
        embedding = embeddings_map.get(paper_id)
        if embedding is None:
            LOGGER.warning("Missing embedding for paper %s", paper_id)
            continue
        embedding_rows.append((embedding.astype("float32").tobytes(), paper_id))

    bulk_save_embeddings(db_path, embedding_rows)

    anchor = max(papers, key=_claim_density)
    anchor_date = _parse_date(anchor.get("pub_date"))
    anchor_authors = set(_parse_json_list(anchor.get("authors")))

    challengers = []
    for paper in papers:
        if paper.get("paper_id") == anchor.get("paper_id"):
            continue
        paper_date = _parse_date(paper.get("pub_date"))
        if anchor_date and paper_date and paper_date <= anchor_date:
            continue
        paper_authors = set(_parse_json_list(paper.get("authors")))
        if paper_authors and anchor_authors and paper_authors.issubset(anchor_authors):
            continue
        challengers.append(paper)

    if len(challengers) < 3:
        LOGGER.warning("Cluster %s skipped due to insufficient challengers", cluster_id)
        set_cluster_status(db_path, cluster_id, "skipped", pair_count=0)
        return

    pairs = []
    for challenger in challengers:
        # Deterministic pair id makes stage2 idempotent across reruns.
        pair_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{cluster_id}:{anchor.get('paper_id')}:{challenger.get('paper_id')}",
            )
        )
        pair = {
            "pair_id": pair_id,
            "cluster_id": cluster_id,
            "anchor_id": anchor.get("paper_id"),
            "challenger_id": challenger.get("paper_id"),
            "similarity_score": None,
            "status": "pending",
        }
        pairs.append(pair)

    bulk_save_pairs(db_path, pairs)

    set_cluster_status(
        db_path, cluster_id, "stage2", pair_count=len(challengers)
    )
    LOGGER.info("Stage 2 complete for %s", cluster_id)
