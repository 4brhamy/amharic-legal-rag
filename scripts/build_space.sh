#!/usr/bin/env bash
# Assemble the Hugging Face Space in dist/space: the Gradio app plus the library and corpus it needs.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/space && mkdir -p dist/space/amlegal dist/space/data
cp space/app.py space/requirements.txt space/README.md dist/space/
cp amlegal/__init__.py amlegal/ethiopic.py amlegal/retrieval.py amlegal/generate.py dist/space/amlegal/
cp data/family_code.jsonl dist/space/data/
echo "Space files:" && find dist/space -type f | sort
