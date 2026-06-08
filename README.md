# analysis_helpers

Utilities for analysis workflows, including optional ROOT-based helpers.

## Checking ROOT availability

ROOT is optional. You can check whether PyROOT is available before calling
ROOT-dependent functions.

```python
from analysis_helpers import is_root_available

if is_root_available():
    # Safe to call ROOT-dependent functions/classes here
    pass
else:
    print("PyROOT is not available in this environment")
```

If you want to fail fast, use:

```python
from analysis_helpers import require_root

require_root()  # Raises ImportError with a clear message when ROOT is missing
```

## Automatic ROOT setup for uv

`uv` does not provide a native post-`sync` hook. This repository includes a
wrapper that adds one practical automation step:

1. Run `uv` command (`sync`, `init`, `run`, ...)
2. Detect `root-config` on the system
3. If ROOT is available, write a `.pth` file in `.venv` so `uv run python`
   can import `ROOT`

Use:

```bash
./scripts/uv-root sync --extra test
./scripts/uv-root run python -c "import ROOT; print(ROOT.__file__)"
```

If you want this behavior everywhere in this repo, you can define an alias:

```bash
alias uv='./scripts/uv-root'
```

