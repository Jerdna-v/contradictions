#!/usr/bin/env python3
"""
Test script: extract claims for a single pair and verify DB persistence.
Designed to run inside the container and diagnose stage4 claim-saving issues.
"""
import os
import sys
import logging
import sqlite3
import json

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_stage4")

# Add contradictions to path
sys.path.insert(0, "/d/hpc/home/an49507/project/contradictions")

from config.settings import (
    SQLITE_DB_PATH,
    PHI3_ENDPOINT,
    PHI3_MODEL_NAME,
    MAX_CLAIMS_PER_PAPER,
)
from db.schema import (
    get_pairs_by_status,
    get_papers_for_cluster,
    save_claims,
    update_pair_status,
)
from pipeline.stage4_claims import _extract_claims
from pipeline.prompt_loader import load_prompt

def main():
    logger.info("=" * 80)
    logger.info("TEST: Single-pair claim extraction and DB verification")
    logger.info("=" * 80)
    
    logger.info(f"DB Path: {SQLITE_DB_PATH}")
    logger.info(f"vLLM Endpoint: {PHI3_ENDPOINT}")
    logger.info(f"vLLM Model: {PHI3_MODEL_NAME}")
    
    # Verify DB exists and is readable
    if not os.path.exists(SQLITE_DB_PATH):
        logger.error(f"DB not found at {SQLITE_DB_PATH}")
        return 1
    
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Tables in DB: {tables}")
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM candidate_pairs;")
        pair_count = cursor.fetchone()[0]
        logger.info(f"Total pairs in DB: {pair_count}")
        
        cursor.execute("SELECT COUNT(*) FROM claims;")
        claim_count = cursor.fetchone()[0]
        logger.info(f"Total claims in DB: {claim_count}")
        
        # Count pairs by status
        cursor.execute("SELECT status, COUNT(*) FROM candidate_pairs GROUP BY status;")
        for status, count in cursor.fetchall():
            logger.info(f"  {status}: {count}")
        
        conn.close()
    except Exception as exc:
        logger.error(f"Failed to query DB: {exc}", exc_info=True)
        return 1
    
    # Find a cluster and get a pair with similarity_passed status
    logger.info("\n" + "=" * 80)
    logger.info("Finding test pair...")
    logger.info("=" * 80)
    
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Get all clusters
        cursor.execute("SELECT DISTINCT cluster_id FROM candidate_pairs LIMIT 1;")
        result = cursor.fetchone()
        if not result:
            logger.error("No clusters found in DB")
            conn.close()
            return 1
        
        cluster_id = result[0]
        logger.info(f"Using cluster: {cluster_id}")
        
        # Get a pair with similarity_passed status
        cursor.execute(
            "SELECT * FROM candidate_pairs WHERE cluster_id=? AND status='similarity_passed' LIMIT 1;",
            (cluster_id,)
        )
        pair_row = cursor.fetchone()
        if not pair_row:
            logger.error(f"No similarity_passed pairs in cluster {cluster_id}")
            conn.close()
            return 1
        
        # Get columns
        cursor.execute("PRAGMA table_info(candidate_pairs);")
        columns = {col[1]: idx for col in cursor.fetchall()}
        pair = {col: pair_row[idx] for col, idx in columns.items()}
        
        logger.info(f"Test pair: {pair}")
        pair_id = pair.get("pair_id")
        anchor_id = pair.get("anchor_id")
        
        # Get papers
        cursor.execute(
            "SELECT * FROM papers WHERE cluster_id=? AND (paper_id=?);",
            (cluster_id, anchor_id)
        )
        paper_row = cursor.fetchone()
        if not paper_row:
            logger.error(f"Anchor paper {anchor_id} not found")
            conn.close()
            return 1
        
        # Get paper columns
        cursor.execute("PRAGMA table_info(papers);")
        paper_columns = {col[1]: idx for col in cursor.fetchall()}
        anchor = {col: paper_row[idx] for col, idx in paper_columns.items()}
        
        logger.info(f"Anchor paper: paper_id={anchor.get('paper_id')}, title={anchor.get('title')[:50] if anchor.get('title') else 'N/A'}...")
        
        conn.close()
    except Exception as exc:
        logger.error(f"Failed to find test pair: {exc}", exc_info=True)
        return 1
    
    # Run extraction
    logger.info("\n" + "=" * 80)
    logger.info("Running claim extraction...")
    logger.info("=" * 80)
    
    try:
        text = " ".join(
            [
                anchor.get("title", ""),
                anchor.get("abstract", ""),
                anchor.get("intro_text", ""),
                anchor.get("results_text", ""),
                anchor.get("conclusion_text", ""),
            ]
        ).strip()
        
        logger.info(f"Paper text length: {len(text)} chars")
        
        prompt_path = "/d/hpc/home/an49507/project/contradictions/prompts/claim_extraction.txt"
        logger.info(f"Loading prompt from: {prompt_path}")
        
        claims = _extract_claims(prompt_path, text)
        logger.info(f"Extracted {len(claims)} claims (before filtering)")
        
        if claims:
            logger.info(f"First claim sample: {claims[0]}")
        
        # Filter as stage4 does
        filtered = []
        seen = set()
        for idx, claim in enumerate(claims):
            claim_text = claim.get("text", "").strip()
            if len(claim_text) < 20:
                logger.debug(f"Skipping claim {idx}: too short ({len(claim_text)} chars)")
                continue
            if claim_text in seen:
                logger.debug(f"Skipping claim {idx}: duplicate")
                continue
            seen.add(claim_text)
            filtered.append(
                {
                    "claim_id": f"TEST_{pair_id}_{idx}",
                    "pair_id": pair_id,
                    "paper_id": anchor.get("paper_id"),
                    "claim_text": claim_text,
                    "source_section": claim.get("section"),
                    "claim_index": idx,
                }
            )
        
        filtered = filtered[:MAX_CLAIMS_PER_PAPER]
        logger.info(f"After filtering: {len(filtered)} claims")
        
        if filtered:
            logger.info(f"Sample filtered claim: {filtered[0]}")
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}", exc_info=True)
        return 1
    
    # Save claims to DB
    logger.info("\n" + "=" * 80)
    logger.info("Saving claims to DB...")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Calling save_claims(db_path={SQLITE_DB_PATH}, {len(filtered)} claims)")
        save_claims(SQLITE_DB_PATH, filtered)
        logger.info("save_claims completed without exception")
    except Exception as exc:
        logger.error(f"save_claims failed: {exc}", exc_info=True)
        return 1
    
    # Verify claims in DB
    logger.info("\n" + "=" * 80)
    logger.info("Verifying claims in DB...")
    logger.info("=" * 80)
    
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # Total claims
        cursor.execute("SELECT COUNT(*) FROM claims;")
        total_claims = cursor.fetchone()[0]
        logger.info(f"Total claims in DB after save: {total_claims}")
        
        # Claims for this pair
        cursor.execute(
            "SELECT COUNT(*) FROM claims WHERE pair_id=?;",
            (pair_id,)
        )
        pair_claims = cursor.fetchone()[0]
        logger.info(f"Claims for test pair {pair_id}: {pair_claims}")
        
        if pair_claims > 0:
            cursor.execute(
                "SELECT claim_id, claim_text, source_section FROM claims WHERE pair_id=? LIMIT 3;",
                (pair_id,)
            )
            for row in cursor.fetchall():
                logger.info(f"  Claim: {row[0]} | {row[1][:60]}... | {row[2]}")
        else:
            logger.warning("No claims found for test pair after save!")
        
        conn.close()
        
        if pair_claims == len(filtered):
            logger.info(f"SUCCESS: All {len(filtered)} claims persisted to DB!")
            return 0
        else:
            logger.error(f"MISMATCH: Expected {len(filtered)} claims, found {pair_claims}")
            return 1
    except Exception as exc:
        logger.error(f"Verification query failed: {exc}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
