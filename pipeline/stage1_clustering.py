import logging
from typing import Dict, List

from config.settings import CSO_BATCH_WORKERS, CSO_DELETE_OUTLIERS, CSO_MIN_CLUSTER_SIZE
from db.schema import (
    bulk_save_cluster_memberships,
    bulk_upsert_papers,
    init_db,
    set_cluster_status,
)
from pipeline.data_loader import load_papers_with_metadata
from pipeline.text_utils import (
    clean_text,
    lexical_filter_enabled,
    load_lexical_dict,
    matches_lexical_dict,
)


LOGGER = logging.getLogger("contradiction_pipeline")


def _build_cso_classifier():
    try:
        from cso_classifier import CSOClassifier
    except ImportError:
        LOGGER.warning("cso-classifier not installed; falling back to metadata categories.")
        return None

    LOGGER.info("Initializing CSO classifier once for stage1.")
    return CSOClassifier(
        modules="both",
        enhancement="first",
        delete_outliers=CSO_DELETE_OUTLIERS,
    )


def _classify_cso(classifier, tags_text: Dict[str, str]) -> List[str]:
    if classifier is None:
        return []

    if hasattr(classifier, "classify_paper"):
        result = classifier.classify_paper(tags_text)
    else:
        result = classifier.run(tags_text)
    return list(result.get("union") or [])


def _classify_cso_batch(classifier, papers_for_cso: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    if classifier is None or not papers_for_cso:
        return {}

    if hasattr(classifier, "batch_run"):
        LOGGER.info(
            "Running CSO batch classification for %d papers with workers=%d.",
            len(papers_for_cso),
            CSO_BATCH_WORKERS,
        )
        raw = classifier.batch_run(papers_for_cso, workers=CSO_BATCH_WORKERS)
        return {paper_id: list((result or {}).get("union") or []) for paper_id, result in raw.items()}

    LOGGER.info("CSO batch API unavailable; falling back to per-paper classification.")
    return {
        paper_id: _classify_cso(classifier, payload)
        for paper_id, payload in papers_for_cso.items()
    }


def _categories_fallback(paper: Dict[str, object]) -> List[str]:
    categories = paper.get("categories") or ""
    if isinstance(categories, list):
        return [c for c in categories if c]
    if isinstance(categories, str):
        return [c for c in categories.split() if c]
    return []


def _apply_lexical_filter(
    paper: Dict[str, object],
    tags: List[str],
    lexical_entries: List[Dict[str, object]],
    enabled: bool,
) -> List[str]:
    if not enabled:
        return tags

    text = clean_text(
        " ".join(
            [
                paper.get("title", ""),
                paper.get("abstract", ""),
                paper.get("intro_text", ""),
            ]
        )
    )

    retained = []
    for tag in tags:
        for entry in lexical_entries:
            if matches_lexical_dict(text, entry):
                retained.append(tag)
                break
    return retained


def run_stage1(db_path: str, metadata_path: str) -> None:
    init_db(db_path)
    papers = load_papers_with_metadata(metadata_path)
    if not papers:
        LOGGER.warning("No papers loaded from metadata.")
        return

    classifier = _build_cso_classifier()

    lexical_enabled = lexical_filter_enabled()
    lexical_entries: List[Dict[str, object]] = []
    if lexical_enabled:
        from config.settings import LEXICAL_DICT_PATH

        lexical_entries = load_lexical_dict(LEXICAL_DICT_PATH)
        LOGGER.info("Lexical filter enabled.")
    else:
        LOGGER.info("Lexical filter disabled - using raw CSO tags.")

    cluster_map = {}

    papers_for_cso = {
        str(idx): {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
        }
        for idx, paper in enumerate(papers)
        if not paper.get("cso_tags")
    }
    cso_results = _classify_cso_batch(classifier, papers_for_cso)
    # Persist in batches so stage1 can be resumed without reprocessing all papers
    BATCH = 1000
    upsert_batch = []
    membership_batch = []

    for idx, paper in enumerate(papers):
        cso_tags = paper.get("cso_tags")
        if not cso_tags:
            cso_tags = cso_results.get(str(idx), [])

        if not cso_tags:
            cso_tags = _categories_fallback(paper)

        cso_tags = _apply_lexical_filter(
            paper,
            cso_tags,
            lexical_entries,
            lexical_enabled,
        )
        paper["cso_tags"] = cso_tags

        for tag in cso_tags:
            cluster_map.setdefault(tag, []).append(paper.get("paper_id"))
            membership_batch.append((tag, paper.get("paper_id")))

        upsert_batch.append(paper)

        # Flush periodically
        if (idx + 1) % BATCH == 0:
            bulk_upsert_papers(db_path, upsert_batch)
            bulk_save_cluster_memberships(db_path, membership_batch)
            upsert_batch = []
            membership_batch = []

        if (idx + 1) % 1000 == 0:
            LOGGER.info("Stage1 progress: processed %d/%d papers", idx + 1, len(papers))

    # Flush remaining
    if upsert_batch:
        bulk_upsert_papers(db_path, upsert_batch)
    if membership_batch:
        bulk_save_cluster_memberships(db_path, membership_batch)

    for cluster_id, members in cluster_map.items():
        if len(members) < CSO_MIN_CLUSTER_SIZE:
            continue
        set_cluster_status(
            db_path, cluster_id, "stage1", paper_count=len(members), pair_count=0
        )

    LOGGER.info("Stage 1 complete. Clusters: %d", len(cluster_map))
