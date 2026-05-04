import json
import numpy as np
import pytest

from db.schema import init_db, load_embedding, save_embedding, upsert_paper
from pipeline.stage3_similarity import cosine_matrix
from pipeline.stage4_claims import _parse_claims
from pipeline.stage5_evidence import extract_evidence
from pipeline.stage6_nli import ensemble_label
from pipeline.text_utils import clean_text, extract_sections, matches_lexical_dict


def test_extract_sections():
    body_text = [
        {"heading": "Introduction", "text": "Intro text. More."},
        {"heading": "Results", "text": "Results here."},
        {"heading": "Conclusion", "text": "Conclusion here."},
        {"heading": "Limitations", "text": "Limitations here."},
        {"heading": "Future Work", "text": "Future work here."},
    ]
    sections = extract_sections(body_text)
    assert "Intro text" in sections["intro_text"]
    assert "Results here" in sections["results_text"]
    assert "Conclusion here" in sections["conclusion_text"]
    assert "Limitations here" in sections["limitations_text"]
    assert "Future work here" in sections["future_work_text"]


def test_clean_text():
    text = "Ref {{cite:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}} figure 1."
    cleaned = clean_text(text)
    assert "cite" not in cleaned
    assert "figure" not in cleaned.lower()


def test_lexical_match():
    entry = {"canonical": "transformer", "aliases": ["attention"]}
    assert matches_lexical_dict("The transformer model", entry)
    assert matches_lexical_dict("Uses attention mechanisms", entry)


def test_cosine_matrix():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrix = cosine_matrix(emb)
    assert np.isclose(matrix[0, 0], 1.0)
    assert np.isclose(matrix[1, 1], 1.0)
    assert np.isclose(matrix[0, 1], 0.0)


def test_claim_json_parsing_valid():
    raw = json.dumps({"claims": [{"text": "Claim A", "section": "results"}]})
    parsed = _parse_claims(raw)
    assert parsed[0]["text"] == "Claim A"


def test_claim_json_parsing_invalid():
    with pytest.raises(json.JSONDecodeError):
        _parse_claims("{bad json")


def test_evidence_null_handling():
    payload = {"evidence": None, "section": None}
    evidence_text, section = extract_evidence(payload)
    assert evidence_text is None
    assert section is None


def test_ensemble_label_logic():
    assert ensemble_label("contradiction", 0.9, "contradiction", 0.7) == "contradiction"
    assert ensemble_label("contradiction", 0.9, "entailment", 0.7) == "flagged"
    assert ensemble_label("entailment", 0.9, "entailment", 0.7) == "support"
    assert ensemble_label("neutral", 0.9, "neutral", 0.7) == "neutral"


def test_embedding_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(str(db_path))
    upsert_paper(
        str(db_path),
        {
            "paper_id": "p1",
            "title": "t",
            "abstract": "a",
            "intro_text": "",
            "results_text": "",
            "conclusion_text": "",
            "limitations_text": "",
            "future_work_text": "",
            "pub_date": "",
            "authors": [],
            "cso_tags": [],
        },
    )
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    save_embedding(str(db_path), "p1", embedding)
    loaded = load_embedding(str(db_path), "p1")
    assert np.allclose(embedding, loaded)
