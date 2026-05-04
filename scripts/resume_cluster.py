import argparse
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _bootstrap_env_from_argv() -> None:
    if "--env" not in sys.argv:
        return
    idx = sys.argv.index("--env")
    if idx + 1 >= len(sys.argv):
        return
    load_dotenv(sys.argv[idx + 1], override=True)


_bootstrap_env_from_argv()

from config.settings import METADATA_PATH, RETRIEVAL_VECTORS_PATH, SQLITE_DB_PATH
from pipeline.stage2_anchor import run_stage2
from pipeline.stage3_similarity import run_stage3
from pipeline.stage4_claims import run_stage4
from pipeline.stage5_evidence import run_stage5
from pipeline.stage6_nli import run_stage6
from pipeline.stage7_typing import run_stage7


def _prompt_paths(root: str) -> dict:
    return {
        "claim_extraction": os.path.join(root, "prompts", "claim_extraction.txt"),
        "evidence_retrieval": os.path.join(root, "prompts", "evidence_retrieval.txt"),
        "nli_zero_shot": os.path.join(root, "prompts", "nli_zero_shot.txt"),
        "typing_fewshot": os.path.join(root, "prompts", "typing_fewshot.txt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--cluster-id", required=True)
    args = parser.parse_args()

    load_dotenv(args.env, override=True)
    prompt_paths = _prompt_paths(os.path.dirname(os.path.dirname(__file__)))

    run_stage2(SQLITE_DB_PATH, args.cluster_id, METADATA_PATH, RETRIEVAL_VECTORS_PATH)
    run_stage3(SQLITE_DB_PATH, args.cluster_id)
    run_stage4(SQLITE_DB_PATH, args.cluster_id, prompt_paths["claim_extraction"])
    run_stage5(SQLITE_DB_PATH, args.cluster_id, prompt_paths["evidence_retrieval"])
    run_stage6(SQLITE_DB_PATH, args.cluster_id, prompt_paths["nli_zero_shot"])
    run_stage7(SQLITE_DB_PATH, args.cluster_id, prompt_paths["typing_fewshot"])


if __name__ == "__main__":
    main()
