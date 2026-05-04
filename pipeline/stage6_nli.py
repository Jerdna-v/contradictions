import asyncio
import json
import logging
import math
import uuid

from config.settings import (
    BLOOMZ_ENDPOINT,
    BLOOMZ_MODEL_NAME,
    LLAMA_ENDPOINT,
    LLAMA_MODEL_NAME,
    NLI_CONFIDENCE_THRESHOLD,
)
from db.schema import (
    get_claims_for_pair,
    get_evidence_for_pair,
    get_pairs_by_status,
    save_nli_result,
    set_cluster_status,
    update_pair_status,
)
from models.vllm_client import complete
from pipeline.prompt_loader import load_prompt

LOGGER = logging.getLogger("contradiction_pipeline")


def _parse_bloomz_label(text: str) -> str:
    token = text.strip().split()[0].lower() if text else ""
    if token.startswith("entail"):
        return "entailment"
    if token.startswith("contrad"):
        return "contradiction"
    if token.startswith("neutral"):
        return "neutral"
    return "neutral"


def _extract_logprob(response: dict) -> float:
    try:
        logprob = response["choices"][0]["logprobs"]["token_logprobs"][0]
        return float(logprob)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _parse_llama_json(text: str) -> dict:
    return json.loads(text)


def ensemble_label(bloomz_label: str, bloomz_score: float, llama_label: str, llama_conf: float) -> str:
    if bloomz_label == "contradiction" and llama_label == "contradiction":
        if llama_conf >= NLI_CONFIDENCE_THRESHOLD:
            return "contradiction"
        return "flagged"
    if bloomz_label == "contradiction" and llama_label != "contradiction":
        return "flagged"
    if llama_label == "contradiction" and bloomz_label != "contradiction":
        return "flagged"
    if bloomz_label == "entailment" and llama_label == "entailment":
        return "support"
    return "neutral"


def run_stage6(db_path: str, cluster_id: str, prompt_path: str) -> None:
    pairs = get_pairs_by_status(db_path, cluster_id, "evidence_retrieved")
    processed = 0

    for pair in pairs:
        claims = get_claims_for_pair(db_path, pair.get("pair_id"))
        evidence = get_evidence_for_pair(db_path, pair.get("pair_id"))
        evidence_map = {e.get("claim_id"): e for e in evidence}

        for claim in claims:
            evidence_row = evidence_map.get(claim.get("claim_id"))
            if not evidence_row:
                continue

            evidence_text = evidence_row.get("evidence_text", "")
            claim_text = claim.get("claim_text", "")

            bloomz_prompt = f"premise: {evidence_text}\nhypothesis: {claim_text}\nrelationship:"
            llama_prompt = load_prompt(prompt_path).format(
                evidence_text=evidence_text, claim_text=claim_text
            )

            async def run_bloomz():
                return await complete(
                    BLOOMZ_ENDPOINT,
                    BLOOMZ_MODEL_NAME,
                    bloomz_prompt,
                    max_tokens=5,
                    temperature=0.0,
                    logprobs=1,
                )

            async def run_llama():
                return await complete(
                    LLAMA_ENDPOINT,
                    LLAMA_MODEL_NAME,
                    llama_prompt,
                    max_tokens=200,
                    temperature=0.0,
                    logprobs=5,
                )

            bloomz_resp, llama_resp = asyncio.run(
                asyncio.gather(run_bloomz(), run_llama())
            )

            bloomz_text = bloomz_resp.get("choices", [{}])[0].get("text", "")
            bloomz_label = _parse_bloomz_label(bloomz_text)
            bloomz_logprob = _extract_logprob(bloomz_resp)
            bloomz_score = math.exp(bloomz_logprob)

            llama_text = llama_resp.get("choices", [{}])[0].get("text", "")
            try:
                llama_payload = _parse_llama_json(llama_text)
                llama_label = llama_payload.get("label", "neutral")
                llama_conf = float(llama_payload.get("confidence", 0.0))
            except json.JSONDecodeError:
                LOGGER.warning("Llama NLI JSON parse failed for claim %s", claim.get("claim_id"))
                llama_label = "neutral"
                llama_conf = 0.0

            label = ensemble_label(bloomz_label, bloomz_score, llama_label, llama_conf)

            save_nli_result(
                db_path,
                {
                    "nli_id": str(uuid.uuid4()),
                    "claim_id": claim.get("claim_id"),
                    "evidence_id": evidence_row.get("evidence_id"),
                    "pair_id": pair.get("pair_id"),
                    "bloomz_label": bloomz_label,
                    "bloomz_score": bloomz_score,
                    "llama_label": llama_label,
                    "llama_confidence": llama_conf,
                    "ensemble_label": label,
                },
            )

        update_pair_status(db_path, pair.get("pair_id"), "nli_done")
        processed += 1
        if processed % 100 == 0:
            set_cluster_status(db_path, cluster_id, "stage6", pair_count=processed)

    set_cluster_status(db_path, cluster_id, "stage6", pair_count=processed)
