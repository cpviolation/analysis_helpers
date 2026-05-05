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

