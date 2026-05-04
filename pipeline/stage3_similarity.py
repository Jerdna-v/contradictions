import logging
from typing import Dict, List, Optional

import numpy as np

from config.settings import SIMILARITY_THRESHOLD
from db.schema import (
    bulk_update_pair_status,
    get_pairs_by_status,
    get_papers_for_cluster,
    load_embedding,
    set_cluster_status,
)

LOGGER = logging.getLogger("contradiction_pipeline")


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T


def run_stage3(db_path: str, cluster_id: str, embedding_cache: Optional[Dict] = None) -> None:
    papers = get_papers_for_cluster(db_path, cluster_id)
    if not papers:
        LOGGER.warning("No papers found for cluster %s", cluster_id)
        return

    paper_ids = []
    embedding_list = []
    for paper in papers:
        paper_id = paper.get("paper_id")
        if embedding_cache is not None:
            embedding = embedding_cache.get(paper_id)
        else:
            embedding = load_embedding(db_path, paper_id)
        if embedding is None:
            LOGGER.warning("Missing embedding for %s", paper_id)
            continue
        paper_ids.append(paper_id)
        embedding_list.append(embedding)

    if not embedding_list:
        LOGGER.warning("No embeddings available for cluster %s", cluster_id)
        return

    matrix = cosine_matrix(np.vstack(embedding_list))
    index_map = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}

    pairs = get_pairs_by_status(db_path, cluster_id, "pending")
    skipped = 0
    updates = []
    for pair in pairs:
        anchor_idx = index_map.get(pair.get("anchor_id"))
        challenger_idx = index_map.get(pair.get("challenger_id"))
        if anchor_idx is None or challenger_idx is None:
            LOGGER.warning("Missing embeddings for pair %s", pair.get("pair_id"))
            updates.append(("skipped", None, pair.get("pair_id")))
            skipped += 1
            continue

        sim = float(matrix[anchor_idx, challenger_idx])
        if sim < SIMILARITY_THRESHOLD:
            updates.append(("skipped", sim, pair.get("pair_id")))
            skipped += 1
        else:
            updates.append(("similarity_passed", sim, pair.get("pair_id")))

    bulk_update_pair_status(db_path, updates)

    set_cluster_status(db_path, cluster_id, "stage3")
    LOGGER.info("Stage 3 complete for %s (skipped %d)", cluster_id, skipped)
