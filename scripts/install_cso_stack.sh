#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ./.venv/bin/python ]]; then
  python3 -m venv .venv
fi

./.venv/bin/python -m pip install "pip<24" "setuptools<60" "wheel<0.40" "numpy<2"
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -r requirements.cso.txt
./.venv/bin/pip install --no-deps cso-classifier==3.1

./.venv/bin/python -m spacy download en_core_web_sm
./.venv/bin/python - <<'PY'
import nltk
nltk.download('stopwords')
PY

echo "CSO stack installed in contradictions/.venv"
