import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

import numpy as np


def _connect(db_path: str) -> sqlite3.Connection:
    timeout_sec = float(os.getenv("SQLITE_TIMEOUT_SEC", "60"))
    busy_timeout_ms = int(os.getenv("SQLITE_BUSY_TIMEOUT_MS", "60000"))

    conn = sqlite3.connect(db_path, timeout=timeout_sec)
    conn.row_factory = sqlite3.Row
    # WAL significantly reduces writer/reader blocking on SQLite.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _retry_on_locked(func):
    retries = int(os.getenv("SQLITE_WRITE_RETRIES", "8"))
    backoff_sec = float(os.getenv("SQLITE_RETRY_BASE_SEC", "0.05"))
    for attempt in range(retries):
        try:
            return func()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == retries - 1:
                raise
            time.sleep(backoff_sec * (2 ** attempt))


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS cluster_progress (
            cluster_id       TEXT PRIMARY KEY,
            status           TEXT NOT NULL,
            paper_count      INTEGER,
            pair_count       INTEGER,
            error_message    TEXT,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS papers (
            paper_id         TEXT PRIMARY KEY,
            title            TEXT,
            abstract         TEXT,
            intro_text       TEXT,
            results_text     TEXT,
            conclusion_text  TEXT,
            limitations_text TEXT,
            future_work_text TEXT,
            pub_date         DATE,
            authors          TEXT,
            cso_tags         TEXT,
            embedding        BLOB
        );

        CREATE TABLE IF NOT EXISTS cluster_membership (
            cluster_id  TEXT,
            paper_id    TEXT,
            PRIMARY KEY (cluster_id, paper_id)
        );

        CREATE TABLE IF NOT EXISTS candidate_pairs (
            pair_id          TEXT PRIMARY KEY,
            cluster_id       TEXT,
            anchor_id        TEXT,
            challenger_id    TEXT,
            similarity_score REAL,
            status           TEXT DEFAULT "pending",
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_id    TEXT PRIMARY KEY,
            pair_id     TEXT,
            paper_id    TEXT,
            claim_text  TEXT,
            source_section TEXT,
            claim_index INTEGER
        );

        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id   TEXT PRIMARY KEY,
            claim_id      TEXT,
            pair_id       TEXT,
            evidence_text TEXT,
            source_section TEXT
        );

        CREATE TABLE IF NOT EXISTS nli_results (
            nli_id           TEXT PRIMARY KEY,
            claim_id         TEXT,
            evidence_id      TEXT,
            pair_id          TEXT,
            bloomz_label     TEXT,
            bloomz_score     REAL,
            llama_label      TEXT,
            llama_confidence REAL,
            ensemble_label   TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS contradictions (
            contradiction_id  TEXT PRIMARY KEY,
            pair_id           TEXT,
            claim_id          TEXT,
            evidence_id       TEXT,
            contradiction_type TEXT,
            explanation       TEXT,
            reviewer_label    TEXT,
            reviewer_notes    TEXT,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    conn.commit()
    conn.close()


def get_cluster_status(db_path: str, cluster_id: str) -> Optional[str]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status FROM cluster_progress WHERE cluster_id = ?",
        (cluster_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return row["status"] if row else None


def set_cluster_status(db_path: str, cluster_id: str, status: str, **kwargs) -> None:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO cluster_progress (cluster_id, status, paper_count, pair_count, error_message)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET
            status = excluded.status,
            paper_count = excluded.paper_count,
            pair_count = excluded.pair_count,
            error_message = excluded.error_message,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            cluster_id,
            status,
            kwargs.get("paper_count"),
            kwargs.get("pair_count"),
            kwargs.get("error_message"),
        ),
    )
    conn.commit()
    conn.close()


def upsert_paper(db_path: str, paper: Dict[str, object]) -> None:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO papers (
            paper_id, title, abstract, intro_text, results_text, conclusion_text,
            limitations_text, future_work_text, pub_date, authors, cso_tags, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT embedding FROM papers WHERE paper_id = ?), NULL))
        """,
        (
            paper.get("paper_id"),
            paper.get("title"),
            paper.get("abstract"),
            paper.get("intro_text"),
            paper.get("results_text"),
            paper.get("conclusion_text"),
            paper.get("limitations_text"),
            paper.get("future_work_text"),
            paper.get("pub_date"),
            json.dumps(paper.get("authors") or []),
            json.dumps(paper.get("cso_tags") or []),
            paper.get("paper_id"),
        ),
    )
    conn.commit()
    conn.close()


def bulk_upsert_papers(db_path: str, papers: List[Dict[str, object]]) -> None:
    if not papers:
        return

    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR REPLACE INTO papers (
            paper_id, title, abstract, intro_text, results_text, conclusion_text,
            limitations_text, future_work_text, pub_date, authors, cso_tags, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT embedding FROM papers WHERE paper_id = ?), NULL))
        """,
        [
            (
                paper.get("paper_id"),
                paper.get("title"),
                paper.get("abstract"),
                paper.get("intro_text"),
                paper.get("results_text"),
                paper.get("conclusion_text"),
                paper.get("limitations_text"),
                paper.get("future_work_text"),
                paper.get("pub_date"),
                json.dumps(paper.get("authors") or []),
                json.dumps(paper.get("cso_tags") or []),
                paper.get("paper_id"),
            )
            for paper in papers
        ],
    )
    conn.commit()
    conn.close()


def get_papers_for_cluster(db_path: str, cluster_id: str) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.* FROM papers p
        JOIN cluster_membership cm ON cm.paper_id = p.paper_id
        WHERE cm.cluster_id = ?
        """,
        (cluster_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append(dict(row))
    return results


def save_embedding(db_path: str, paper_id: str, embedding: np.ndarray) -> None:
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE papers SET embedding = ? WHERE paper_id = ?",
            (embedding.astype(np.float32).tobytes(), paper_id),
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def bulk_save_embeddings(db_path: str, rows: List[tuple]) -> None:
    if not rows:
        return

    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE papers SET embedding = ? WHERE paper_id = ?",
            rows,
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def load_embedding(db_path: str, paper_id: str) -> Optional[np.ndarray]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT embedding FROM papers WHERE paper_id = ?",
        (paper_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row or row["embedding"] is None:
        return None
    return np.frombuffer(row["embedding"], dtype=np.float32)


def save_cluster_membership(db_path: str, cluster_id: str, paper_id: str) -> None:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO cluster_membership (cluster_id, paper_id) VALUES (?, ?)",
        (cluster_id, paper_id),
    )
    conn.commit()
    conn.close()


def bulk_save_cluster_memberships(db_path: str, rows: List[tuple]) -> None:
    if not rows:
        return

    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO cluster_membership (cluster_id, paper_id) VALUES (?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def save_pair(db_path: str, pair: Dict[str, object]) -> None:
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO candidate_pairs (
                pair_id, cluster_id, anchor_id, challenger_id, similarity_score, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                pair.get("pair_id"),
                pair.get("cluster_id"),
                pair.get("anchor_id"),
                pair.get("challenger_id"),
                pair.get("similarity_score"),
                pair.get("status", "pending"),
            ),
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def bulk_save_pairs(db_path: str, pairs: List[Dict[str, object]]) -> None:
    if not pairs:
        return

    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR IGNORE INTO candidate_pairs (
                pair_id, cluster_id, anchor_id, challenger_id, similarity_score, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    pair.get("pair_id"),
                    pair.get("cluster_id"),
                    pair.get("anchor_id"),
                    pair.get("challenger_id"),
                    pair.get("similarity_score"),
                    pair.get("status", "pending"),
                )
                for pair in pairs
            ],
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def get_pending_pairs(db_path: str, cluster_id: str) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM candidate_pairs WHERE cluster_id = ? AND status = 'pending'",
        (cluster_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_pairs_by_status(db_path: str, cluster_id: str, status: str) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM candidate_pairs WHERE cluster_id = ? AND status = ?",
        (cluster_id, status),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_pair_status(db_path: str, pair_id: str, status: str, similarity_score: float = None) -> None:
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE candidate_pairs SET status = ?, similarity_score = COALESCE(?, similarity_score) WHERE pair_id = ?",
            (status, similarity_score, pair_id),
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def bulk_update_pair_status(db_path: str, updates: List[tuple]) -> None:
    if not updates:
        return

    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE candidate_pairs SET status = ?, similarity_score = COALESCE(?, similarity_score) WHERE pair_id = ?",
            updates,
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def save_claims(db_path: str, claims: List[Dict[str, object]]) -> None:
    if not claims:
        return
    
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO claims (
                claim_id, pair_id, paper_id, claim_text, source_section, claim_index
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.get("claim_id"),
                    c.get("pair_id"),
                    c.get("paper_id"),
                    c.get("claim_text"),
                    c.get("source_section"),
                    c.get("claim_index"),
                )
                for c in claims
            ],
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def save_evidence(db_path: str, evidence_list: List[Dict[str, object]]) -> None:
    if not evidence_list:
        return
    
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO evidence (
                evidence_id, claim_id, pair_id, evidence_text, source_section
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    e.get("evidence_id"),
                    e.get("claim_id"),
                    e.get("pair_id"),
                    e.get("evidence_text"),
                    e.get("source_section"),
                )
                for e in evidence_list
            ],
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def save_nli_result(db_path: str, nli: Dict[str, object]) -> None:
    conn = _connect(db_path)
    
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO nli_results (
                nli_id, claim_id, evidence_id, pair_id, bloomz_label, bloomz_score,
                llama_label, llama_confidence, ensemble_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nli.get("nli_id"),
                nli.get("claim_id"),
                nli.get("evidence_id"),
                nli.get("pair_id"),
                nli.get("bloomz_label"),
                nli.get("bloomz_score"),
                nli.get("llama_label"),
                nli.get("llama_confidence"),
                nli.get("ensemble_label"),
            ),
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def save_contradiction(db_path: str, contradiction: Dict[str, object]) -> None:
    conn = _connect(db_path)
    
    def _op():
        conn = _connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO contradictions (
                contradiction_id, pair_id, claim_id, evidence_id,
                contradiction_type, explanation, reviewer_label, reviewer_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contradiction.get("contradiction_id"),
                contradiction.get("pair_id"),
                contradiction.get("claim_id"),
                contradiction.get("evidence_id"),
                contradiction.get("contradiction_type"),
                contradiction.get("explanation"),
                contradiction.get("reviewer_label"),
                contradiction.get("reviewer_notes"),
            ),
        )
        conn.commit()
        conn.close()

    _retry_on_locked(_op)


def get_claims_for_pair(db_path: str, pair_id: str) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM claims WHERE pair_id = ?",
        (pair_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_evidence_for_pair(db_path: str, pair_id: str) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM evidence WHERE pair_id = ?",
        (pair_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_nli_for_pair(db_path: str, pair_id: str, label: str = None) -> List[Dict[str, object]]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    if label:
        cursor.execute(
            "SELECT * FROM nli_results WHERE pair_id = ? AND ensemble_label = ?",
            (pair_id, label),
        )
    else:
        cursor.execute(
            "SELECT * FROM nli_results WHERE pair_id = ?",
            (pair_id,),
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_clusters(db_path: str) -> List[str]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT cluster_id FROM cluster_progress")
    rows = cursor.fetchall()
    conn.close()
    return [row["cluster_id"] for row in rows]


def get_pending_clusters(db_path: str) -> List[str]:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT cluster_id FROM cluster_progress WHERE status != 'done'")
    rows = cursor.fetchall()
    conn.close()
    return [row["cluster_id"] for row in rows]


def clusters_exist(db_path: str) -> bool:
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM cluster_progress LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row is not None
