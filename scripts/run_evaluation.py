import argparse
import os
import sqlite3
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

from config.settings import GOLD_STANDARD_PATH, SQLITE_DB_PATH
from evaluation.gold_standard import evaluation_available, load_gold_standard
from evaluation.metrics import bootstrap_ci, macro_f1, precision_recall_f1, mcnemar_test


def _fetch_labels(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT anchor_id, challenger_id, ensemble_label, bloomz_label, llama_label FROM nli_results JOIN candidate_pairs ON nli_results.pair_id = candidate_pairs.pair_id")
    rows = cursor.fetchall()
    conn.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    args = parser.parse_args()

    load_dotenv(args.env, override=True)

    if not evaluation_available():
        print("No gold standard found. Set GOLD_STANDARD_PATH in .env to enable evaluation.")
        print("Pipeline results are available in the SQLite database at:", SQLITE_DB_PATH)
        sys.exit(0)

    gold = load_gold_standard()
    label_map = {(row[0], row[1]): row[2] for row in _fetch_labels(SQLITE_DB_PATH)}

    y_true = []
    y_pred = []
    for row in gold:
        key = (row.get("anchor_id"), row.get("challenger_id"))
        if key not in label_map:
            continue
        y_true.append(row.get("label"))
        y_pred.append(label_map[key])

    metrics = precision_recall_f1(y_true, y_pred)
    mean, low, high = bootstrap_ci(y_true, y_pred)
    print("Binary contradiction detection:")
    print("Precision:", metrics["precision"], "Recall:", metrics["recall"], "F1:", metrics["f1"])
    print("Bootstrap CI (F1):", mean, low, high)
    print("Macro F1:", macro_f1(y_true, y_pred))

    # McNemar test across Bloomz-only vs Llama-only
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT anchor_id, challenger_id, bloomz_label, llama_label FROM nli_results JOIN candidate_pairs ON nli_results.pair_id = candidate_pairs.pair_id")
    rows = cursor.fetchall()
    conn.close()

    bloomz_map = {(r[0], r[1]): r[2] for r in rows}
    llama_map = {(r[0], r[1]): r[3] for r in rows}

    y_bloomz = [bloomz_map.get(k) for k in label_map.keys()]
    y_llama = [llama_map.get(k) for k in label_map.keys()]

    print("McNemar p-value (Bloomz vs Llama):", mcnemar_test(y_true, y_bloomz, y_llama))


if __name__ == "__main__":
    main()
