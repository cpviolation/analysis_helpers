#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
uv sync --extra test
source "$SCRIPT_DIR/.venv/bin/activate"

pytest "$SCRIPT_DIR/tests" --cov=analysis_helpers --cov-report=term-missing --cov-report=xml