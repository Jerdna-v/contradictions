import sqlite3
from typing import Tuple


def load_claim_and_evidence(db_path: str, claim_id: str, evidence_id: str) -> Tuple[str, str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT claim_text FROM claims WHERE claim_id = ?", (claim_id,))
    claim_row = cursor.fetchone()
    cursor.execute("SELECT evidence_text FROM evidence WHERE evidence_id = ?", (evidence_id,))
    evidence_row = cursor.fetchone()
    conn.close()

    claim_text = claim_row[0] if claim_row else ""
    evidence_text = evidence_row[0] if evidence_row else ""
    return claim_text, evidence_text
