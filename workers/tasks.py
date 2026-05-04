import logging

from config.settings import CELERY_RETRY_BACKOFF, CELERY_TASK_RETRIES
from db.schema import set_cluster_status
from pipeline.stage2_anchor import run_stage2
from pipeline.stage3_similarity import run_stage3
from pipeline.stage4_claims import run_stage4
from pipeline.stage5_evidence import run_stage5
from pipeline.stage6_nli import run_stage6
from pipeline.stage7_typing import run_stage7
from workers.celery_app import app

LOGGER = logging.getLogger("contradiction_pipeline")


def process_cluster_sequential(
    db_path: str,
    cluster_id: str,
    metadata_path: str,
    vectors_path: str,
    prompt_paths: dict,
) -> None:
    run_stage2(db_path, cluster_id, metadata_path, vectors_path)
    run_stage3(db_path, cluster_id)
    run_stage4(db_path, cluster_id, prompt_paths["claim_extraction"])
    run_stage5(db_path, cluster_id, prompt_paths["evidence_retrieval"])
    run_stage6(db_path, cluster_id, prompt_paths["nli_zero_shot"])
    run_stage7(db_path, cluster_id, prompt_paths["typing_fewshot"])
    set_cluster_status(db_path, cluster_id, "done")


@app.task(bind=True, max_retries=CELERY_TASK_RETRIES, default_retry_delay=CELERY_RETRY_BACKOFF)
def process_cluster(self, db_path: str, cluster_id: str, metadata_path: str, vectors_path: str, prompt_paths: dict):
    try:
        process_cluster_sequential(db_path, cluster_id, metadata_path, vectors_path, prompt_paths)
    except Exception as exc:
        set_cluster_status(db_path, cluster_id, "error", error_message=str(exc))
        raise self.retry(exc=exc)
