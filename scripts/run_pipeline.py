import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _bootstrap_env_from_argv() -> None:
    if "--env" not in sys.argv:
        return
    idx = sys.argv.index("--env")
    if idx + 1 >= len(sys.argv):
        return
    # Do not override already-exported shell vars (e.g. USE_CELERY=false).
    load_dotenv(sys.argv[idx + 1], override=False)


_bootstrap_env_from_argv()

from config.logging import setup_logging
from config.settings import (
    METADATA_PATH,
    RETRIEVAL_VECTORS_PATH,
    SQLITE_DB_PATH,
)
from db.schema import clusters_exist, get_cluster_status, get_pending_clusters, init_db
from pipeline.stage1_clustering import run_stage1
from pipeline.stage2_anchor import run_stage2
from pipeline.stage3_similarity import run_stage3
from pipeline.stage4_claims import run_stage4
from pipeline.stage5_evidence import run_stage5
from pipeline.stage6_nli import run_stage6
from pipeline.stage7_typing import run_stage7
from workers.tasks import process_cluster


def _parse_stages(value: str):
    if not value:
        return {1, 2, 3, 4, 5, 6, 7}
    return {int(v.strip()) for v in value.split(",") if v.strip()}


def _prompt_paths(root: str) -> dict:
    return {
        "claim_extraction": os.path.join(root, "prompts", "claim_extraction.txt"),
        "evidence_retrieval": os.path.join(root, "prompts", "evidence_retrieval.txt"),
        "nli_zero_shot": os.path.join(root, "prompts", "nli_zero_shot.txt"),
        "typing_fewshot": os.path.join(root, "prompts", "typing_fewshot.txt"),
    }


def _status_to_stage_index(status: str) -> int:
    if not status:
        return 0
    if status in {"done", "skipped"}:
        return 7
    if status.startswith("stage"):
        try:
            return int(status.replace("stage", ""))
        except ValueError:
            return 0
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-id", default=None)
    args = parser.parse_args()

    # Preserve explicit shell env vars while still loading missing values from .env.
    load_dotenv(args.env, override=False)

    logger = setup_logging(os.getenv("LOG_PATH", "logs/pipeline.log"))
    init_db(SQLITE_DB_PATH)

    stages = _parse_stages(args.stages)
    prompt_paths = _prompt_paths(os.path.dirname(os.path.dirname(__file__)))

    if 1 in stages and not clusters_exist(SQLITE_DB_PATH):
        logger.info("Running stage 1")
        run_stage1(SQLITE_DB_PATH, METADATA_PATH)

    cluster_ids = [args.cluster_id] if args.cluster_id else get_pending_clusters(SQLITE_DB_PATH)

    use_celery = os.getenv("USE_CELERY", "true").lower() == "true"
    if stages == {2, 3, 4, 5, 6, 7} and use_celery and not args.cluster_id:
        for cluster_id in cluster_ids:
            process_cluster.delay(
                SQLITE_DB_PATH,
                cluster_id,
                METADATA_PATH,
                RETRIEVAL_VECTORS_PATH,
                prompt_paths,
            )
        logger.info("Dispatched %d cluster tasks to Celery", len(cluster_ids))
        return

    # Pre-load embeddings for stages 2-3 (parallel workers)
    from pipeline.data_loader import load_embeddings_map
    embeddings_map = None
    embedding_cache = None
    num_workers = int(os.getenv("PARALLEL_WORKERS", "4"))
    
    def run_stages_2_3(cluster_id: str, stages_to_run: set) -> None:
        current_status = get_cluster_status(SQLITE_DB_PATH, cluster_id)
        completed_up_to = _status_to_stage_index(current_status)
        
        if 2 in stages_to_run:
            if completed_up_to < 2:
                run_stage2(SQLITE_DB_PATH, cluster_id, METADATA_PATH, RETRIEVAL_VECTORS_PATH, embeddings_map)
                completed_up_to = 2
        if 3 in stages_to_run:
            if completed_up_to < 3:
                run_stage3(SQLITE_DB_PATH, cluster_id, embedding_cache)
                completed_up_to = 3
    
    def run_stages_4_plus(cluster_id: str, stages_to_run: set) -> None:
        current_status = get_cluster_status(SQLITE_DB_PATH, cluster_id)
        completed_up_to = _status_to_stage_index(current_status)

        # Only clusters that passed stage3 should proceed to stage4+.
        if completed_up_to < 3:
            return
        
        if 4 in stages_to_run:
            if completed_up_to < 4:
                run_stage4(SQLITE_DB_PATH, cluster_id, prompt_paths["claim_extraction"])
                completed_up_to = 4
        if 5 in stages_to_run:
            if completed_up_to < 5:
                run_stage5(SQLITE_DB_PATH, cluster_id, prompt_paths["evidence_retrieval"])
                completed_up_to = 5
        if 6 in stages_to_run:
            if completed_up_to < 6:
                run_stage6(SQLITE_DB_PATH, cluster_id, prompt_paths["nli_zero_shot"])
                completed_up_to = 6
        if 7 in stages_to_run:
            if completed_up_to < 7:
                run_stage7(SQLITE_DB_PATH, cluster_id, prompt_paths["typing_fewshot"])

    # Run stages 2-3 in parallel if requested
    if any(s in stages for s in [2, 3]):
        logger.info("Pre-loading embeddings for parallel stage 2-3 processing (%d workers)", num_workers)
        embeddings_map = load_embeddings_map(METADATA_PATH, RETRIEVAL_VECTORS_PATH)
        embedding_cache = embeddings_map  # Use same map for cache
        
        stages_2_3 = {s for s in stages if s in [2, 3]}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_stages_2_3, cid, stages_2_3) for cid in cluster_ids]
            for future in futures:
                future.result()
        logger.info("Stages 2-3 parallel processing complete")

    # Run stages 4+ serially (dependencies)
    if any(s in stages for s in [4, 5, 6, 7]):
        stages_4_plus_set = {s for s in stages if s in [4, 5, 6, 7]}
        for cluster_id in cluster_ids:
            run_stages_4_plus(cluster_id, stages_4_plus_set)


if __name__ == "__main__":
    main()
