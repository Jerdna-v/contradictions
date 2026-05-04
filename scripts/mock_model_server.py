#!/usr/bin/env python3
"""Lightweight HTTP mock that emulates an OpenAI-compatible vLLM completion endpoint.

Returns JSON: {"choices": [{"text": "<json-string-with-claims>"}]}
Accepts any POST path (e.g., /v1/completions).
"""
import argparse
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG = logging.getLogger("mock_model")


def _extract_segment(prompt: str, marker: str) -> str:
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    return prompt[idx + len(marker) :].strip()


def _split_sentences(text: str):
    # Keep sentence splitting simple and deterministic for mock behavior.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _word_chunks(text: str, chunk_words: int = 22):
    words = [w for w in text.split() if w]
    chunks = []
    for i in range(0, len(words), chunk_words):
        part = " ".join(words[i : i + chunk_words]).strip()
        if len(part) >= 30:
            chunks.append(part)
    return chunks


def _ensure_minimum_claims(candidates, text: str, min_claims: int):
    if len(candidates) >= min_claims:
        return candidates

    words = [w for w in text.split() if w]
    if not words:
        words = ["mock", "claim", "text", "placeholder"]

    i = 0
    while len(candidates) < min_claims:
        start = (i * 10) % max(1, len(words))
        snippet_words = words[start : start + 16]
        if not snippet_words:
            snippet_words = words[:16]
        snippet = " ".join(snippet_words).strip()
        if len(snippet) < 20:
            snippet = f"Mock claim {len(candidates)+1} derived from available paper content."
        candidates.append({"text": snippet[:500], "section": "results"})
        i += 1

    return candidates


def _mock_claims(prompt: str) -> dict:
    text = _extract_segment(prompt, "Paper text:")
    max_claims_match = re.search(r"Maximum\s+(\d+)\s+claims", prompt, flags=re.IGNORECASE)
    max_claims = int(max_claims_match.group(1)) if max_claims_match else 5
    max_claims = max(1, min(max_claims, 15))

    candidates = []
    seen = set()
    raw_units = _split_sentences(text)
    if len(raw_units) <= 1:
        # Some preprocessed inputs are one giant sentence; break further by
        # line/semicolon/comma boundaries first, then by fixed word chunks.
        clause_units = [
            u.strip()
            for u in re.split(r"[\n;]+|,(?=\s)\s*", text)
            if u.strip()
        ]
        raw_units = clause_units if len(clause_units) > 1 else _word_chunks(text)

    for sent in raw_units:
        if len(sent) < 30:
            continue
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)

        section = "results"
        low = sent.lower()
        if "conclusion" in low:
            section = "conclusion"
        elif "introduction" in low or "background" in low:
            section = "introduction"

        candidates.append({"text": sent[:500], "section": section})
        if len(candidates) >= max_claims:
            break

    min_claims = min(5, max_claims)
    candidates = _ensure_minimum_claims(candidates, text, min_claims)

    return {"claims": candidates}


def _mock_evidence(prompt: str) -> dict:
    pool = _extract_segment(prompt, "Text from Paper B:")
    sentences = _split_sentences(pool)
    if not sentences:
        return {"evidence": None, "section": None}
    return {"evidence": sentences[0][:500], "section": "conclusion"}


def _mock_nli() -> dict:
    return {"label": "neutral", "confidence": 0.71, "reasoning": "Mock NLI output."}


def _mock_typing() -> dict:
    return {"type": "interpretive", "explanation": "Mock contradiction type output."}


class MockHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        prompt = str(payload.get("prompt", ""))
        prompt_lower = prompt.lower()

        if "scientific claim extractor" in prompt_lower or '"claims"' in prompt_lower:
            out = _mock_claims(prompt)
        elif "text from paper b" in prompt_lower and '"evidence"' in prompt_lower:
            out = _mock_evidence(prompt)
        elif "logical relationship" in prompt_lower and '"label"' in prompt_lower:
            out = _mock_nli()
        elif "contradiction types" in prompt_lower and '"type"' in prompt_lower:
            out = _mock_typing()
        else:
            out = _mock_claims(prompt)

        resp = {"choices": [{"text": json.dumps(out)}]}
        LOG.info(
            "Incoming request path=%s model=%s prompt_len=%d",
            self.path,
            payload.get("model"),
            len(prompt),
        )
        self._send_json(resp)

    def log_message(self, format, *args):
        LOG.info("%s - - %s", self.client_address[0], format % args)


def run_server(port: int, host: str = "0.0.0.0"):
    logging.basicConfig(level=logging.INFO)
    server = HTTPServer((host, port), MockHandler)
    LOG.info("Mock model server starting on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down mock server on port %d", port)
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
