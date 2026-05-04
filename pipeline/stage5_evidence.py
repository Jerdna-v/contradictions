import asyncio
import json
import logging
import uuid

from config.settings import LLAMA_ENDPOINT, LLAMA_MODEL_NAME
from db.schema import (
    get_claims_for_pair,
    get_pairs_by_status,
    get_papers_for_cluster,
    save_evidence,
    set_cluster_status,
    update_pair_status,
)
from models.vllm_client import complete
from pipeline.prompt_loader import load_prompt

LOGGER = logging.getLogger("contradiction_pipeline")


def _parse_evidence(raw: str) -> dict:
    payload = json.loads(raw)
    return payload


def extract_evidence(payload: dict) -> tuple:
    evidence_text = payload.get("evidence")
    section = payload.get("section")
    if evidence_text is None:
        return None, None
    return evidence_text, section


def run_stage5(db_path: str, cluster_id: str, prompt_path: str) -> None:
    pairs = get_pairs_by_status(db_path, cluster_id, "claims_extracted")
    papers = {p.get("paper_id"): p for p in get_papers_for_cluster(db_path, cluster_id)}
    processed = 0

    for pair in pairs:
        challenger = papers.get(pair.get("challenger_id"))
        if not challenger:
            update_pair_status(db_path, pair.get("pair_id"), "evidence_failed")
            continue

        evidence_pool = " ".join(
            [
                challenger.get("conclusion_text", ""),
                challenger.get("limitations_text", ""),
                challenger.get("future_work_text", ""),
            ]
        ).strip()

        evidence_rows = []
        for claim in get_claims_for_pair(db_path, pair.get("pair_id")):
            prompt = load_prompt(prompt_path).format(
                claim_text=claim.get("claim_text"), evidence_pool=evidence_pool
            )
            response = asyncio.run(complete(LLAMA_ENDPOINT, LLAMA_MODEL_NAME, prompt))
            text = response.get("choices", [{}])[0].get("text", "")
            try:
                payload = _parse_evidence(text)
            except json.JSONDecodeError:
                LOGGER.warning("Evidence JSON parse failed for claim %s", claim.get("claim_id"))
                continue

            evidence_text, section = extract_evidence(payload)
            if evidence_text is None:
                LOGGER.warning("No evidence for claim %s", claim.get("claim_id"))
                continue

            evidence_rows.append(
                {
                    "evidence_id": str(uuid.uuid4()),
                    "claim_id": claim.get("claim_id"),
                    "pair_id": pair.get("pair_id"),
                    "evidence_text": evidence_text,
                    "source_section": section,
                }
            )

        save_evidence(db_path, evidence_rows)
        update_pair_status(db_path, pair.get("pair_id"), "evidence_retrieved")
        processed += 1
        if processed % 100 == 0:
            set_cluster_status(db_path, cluster_id, "stage5", pair_count=processed)

    set_cluster_status(db_path, cluster_id, "stage5", pair_count=processed)
