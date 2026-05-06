#!/bin/bash
source .venv/bin/activate && pytest tests --cov=analysis_helpers --cov-report=term-missing --cov-report=xml