#!/usr/bin/env bash
# One command: tests with the coverage floor, the example map, a build, and a smoke install of the wheel.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen
uv run pytest -q
uv run rdlab map check examples/t10-pilot-a.map.yaml
rm -rf dist
uv build
tmp="$(mktemp -d)"
uv venv -q "$tmp/venv"
uv pip install -q --python "$tmp/venv/bin/python" dist/*.whl 2>/dev/null || uv pip install -q --python "$tmp/venv/Scripts/python.exe" dist/*.whl
"$tmp/venv/bin/rdlab" --version 2>/dev/null || "$tmp/venv/Scripts/rdlab.exe" --version
echo "verify: OK"
