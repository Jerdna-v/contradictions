import asyncio
import json
import logging
import uuid
from typing import Dict, List

from config.settings import MAX_CLAIMS_PER_PAPER, PHI3_ENDPOINT, PHI3_MODEL_NAME
from db.schema import (
    get_pairs_by_status,
    get_papers_for_cluster,
    save_claims,
    set_cluster_status,
    update_pair_status,
)
from models.vllm_client import complete
from pipeline.prompt_loader import load_prompt

LOGGER = logging.getLogger("contradiction_pipeline")


def _coerce_json(obj):
    # Accept dict/list or JSON string; attempt to un-wrap double-encoded JSON
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return obj
    if not isinstance(obj, str):
        return None

    text = obj.strip()
    
    # Strip markdown code fences and "Solution N:" prefixes
    import re
    # Remove "Solution N:" prefix if present
    text = re.sub(r"^\s*Solution\s+\d+:\s*", "", text, flags=re.IGNORECASE)
    # Remove markdown code fences (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    
    # Try multiple decode attempts to handle double-encoding
    for attempt in range(3):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                return parsed
            # If parsed is a primitive, try to treat it as the new text
            text = str(parsed)
        except Exception:
            # Try to extract a JSON substring if present
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            break
    return None


def _parse_claims(raw: str) -> List[Dict[str, str]]:
    parsed = _coerce_json(raw)
    if parsed is None:
        return []

    # If the top-level is a list, try to find an object with 'claims'
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "claims" in item:
                parsed = item
                break

    if isinstance(parsed, dict):
        claims = parsed.get("claims", [])
    else:
        claims = []

    return [c for c in claims if isinstance(c, dict) and c.get("text") and c.get("section")]


def _extract_claims(prompt_path: str, paper_text: str) -> List[Dict[str, str]]:
    prompt = load_prompt(prompt_path).format(
        MAX_CLAIMS=MAX_CLAIMS_PER_PAPER, paper_text=paper_text
    )

    # Call the completion endpoint and defensively extract the text field.
    for attempt in range(2):
        try:
            response = asyncio.run(complete(PHI3_ENDPOINT, PHI3_MODEL_NAME, prompt))
        except Exception as exc:
            LOGGER.warning("Model call failed on attempt %d: %s", attempt + 1, exc)
            response = None

        text = ""
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    text = first.get("text", "") or first.get("message", "")
                else:
                    text = str(first)

        # If there's no text at all, log and retry once
        if not text:
            LOGGER.warning("Empty model output on attempt %d", attempt + 1)
            continue

        claims = _parse_claims(text)
        if claims:
            return claims

        # If parsing returned empty, log text snippet for debugging then retry
        LOGGER.warning(
            "Claim parse returned empty on attempt %d; text snippet: %s",
            attempt + 1,
            (text[:500] + "...") if len(text) > 500 else text,
        )

    LOGGER.error("Claim extraction failed after retries; skipping.")
    return []


def run_stage4(db_path: str, cluster_id: str, prompt_path: str) -> None:
    pairs = get_pairs_by_status(db_path, cluster_id, "similarity_passed")
    papers = {p.get("paper_id"): p for p in get_papers_for_cluster(db_path, cluster_id)}
    processed = 0

    for pair in pairs:
        try:
            anchor = papers.get(pair.get("anchor_id"))
            if not anchor:
                update_pair_status(db_path, pair.get("pair_id"), "claims_failed")
                continue

            text = " ".join(
                [
                    anchor.get("title", ""),
                    anchor.get("abstract", ""),
                    anchor.get("intro_text", ""),
                    anchor.get("results_text", ""),
                    anchor.get("conclusion_text", ""),
                ]
            ).strip()

            claims = _extract_claims(prompt_path, text)
            filtered = []
            seen = set()
            for idx, claim in enumerate(claims):
                claim_text = claim.get("text", "").strip()
                if len(claim_text) < 20:
                    continue
                if claim_text in seen:
                    continue
                seen.add(claim_text)
                filtered.append(
                    {
                        "claim_id": str(uuid.uuid4()),
                        "pair_id": pair.get("pair_id"),
                        "paper_id": anchor.get("paper_id"),
                        "claim_text": claim_text,
                        "source_section": claim.get("section"),
                        "claim_index": idx,
                    }
                )

            filtered = filtered[:MAX_CLAIMS_PER_PAPER]
            LOGGER.debug("Stage4: db_path=%s, pair_id=%s, filtered_len=%d", db_path, pair.get("pair_id"), len(filtered))
            if len(filtered) == 0:
                LOGGER.info("No claims to save for pair %s", pair.get("pair_id"))
            try:
                save_claims(db_path, filtered)
                LOGGER.info("save_claims succeeded for pair %s, saved=%d", pair.get("pair_id"), len(filtered))
            except Exception as exc:
                LOGGER.exception("save_claims failed for pair %s: %s", pair.get("pair_id"), exc)
                update_pair_status(db_path, pair.get("pair_id"), "claims_failed")
                continue
            update_pair_status(db_path, pair.get("pair_id"), "claims_extracted")
            LOGGER.info(
                "Claims extracted for pair %s: %d",
                pair.get("pair_id"),
                len(filtered),
            )
        except Exception as exc:
            LOGGER.error("Stage 4 failed for pair %s: %s", pair.get("pair_id"), exc)
            update_pair_status(db_path, pair.get("pair_id"), "claims_failed")
        finally:
            processed += 1
            if processed % 100 == 0:
                set_cluster_status(db_path, cluster_id, "stage4", pair_count=processed)

    set_cluster_status(db_path, cluster_id, "stage4", pair_count=processed)
