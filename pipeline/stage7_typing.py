import asyncio
import json
import logging
import random
import uuid

from config.settings import LLAMA_ENDPOINT, LLAMA_MODEL_NAME, QWEN_ENDPOINT, QWEN_MODEL_NAME
from db.schema import (
    get_nli_for_pair,
    get_pairs_by_status,
    save_contradiction,
    set_cluster_status,
    update_pair_status,
)
from models.vllm_client import complete
from pipeline.prompt_loader import load_prompt
from pipeline.stage7_utils import load_claim_and_evidence

LOGGER = logging.getLogger("contradiction_pipeline")


def run_stage7(db_path: str, cluster_id: str, prompt_path: str, qwen_sample_rate: float = 0.2) -> None:
    pairs = get_pairs_by_status(db_path, cluster_id, "nli_done")
    processed = 0

    for pair in pairs:
        nli_rows = get_nli_for_pair(db_path, pair.get("pair_id"), label="contradiction")
        for nli in nli_rows:
            claim_text, evidence_text = load_claim_and_evidence(
                db_path, nli.get("claim_id"), nli.get("evidence_id")
            )
            prompt = load_prompt(prompt_path).format(
                few_shot_examples="TODO: add few-shot examples",
                claim_text=claim_text,
                evidence_text=evidence_text,
            )
            response = asyncio.run(complete(LLAMA_ENDPOINT, LLAMA_MODEL_NAME, prompt))
            text = response.get("choices", [{}])[0].get("text", "")

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                LOGGER.warning("Typing JSON parse failed for pair %s", pair.get("pair_id"))
                continue

            contradiction = {
                "contradiction_id": str(uuid.uuid4()),
                "pair_id": pair.get("pair_id"),
                "claim_id": nli.get("claim_id"),
                "evidence_id": nli.get("evidence_id"),
                "contradiction_type": payload.get("type"),
                "explanation": payload.get("explanation"),
                "reviewer_label": None,
                "reviewer_notes": None,
            }

            if random.random() < qwen_sample_rate:
                qwen_prompt = (
                    "Classify the contradiction type and explain briefly.\n\n"
                    f"Claim: {claim_text}\nEvidence: {evidence_text}\n"
                    "Return JSON: {\"type\": \"...\", \"notes\": \"...\"}"
                )
                qwen_resp = asyncio.run(
                    complete(QWEN_ENDPOINT, QWEN_MODEL_NAME, qwen_prompt, max_tokens=200)
                )
                qwen_text = qwen_resp.get("choices", [{}])[0].get("text", "")
                try:
                    qwen_payload = json.loads(qwen_text)
                    contradiction["reviewer_label"] = qwen_payload.get("type")
                    contradiction["reviewer_notes"] = qwen_payload.get("notes")
                except json.JSONDecodeError:
                    LOGGER.warning("Qwen JSON parse failed for pair %s", pair.get("pair_id"))

            save_contradiction(db_path, contradiction)

        update_pair_status(db_path, pair.get("pair_id"), "typed")
        processed += 1
        if processed % 100 == 0:
            set_cluster_status(db_path, cluster_id, "stage7", pair_count=processed)

    set_cluster_status(db_path, cluster_id, "done", pair_count=processed)
