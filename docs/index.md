# analysis_helpers

Utilities for HEP analysis workflows, including optional PyROOT-based helpers.

## Installation

<!-- ```bash
pip install analysis_helpers
``` -->
```bash
pip install git+ssh://github.com/cpviolation/analysis_helpers.git
```

To install with documentation dependencies:

<!-- ```bash
pip install analysis_helpers[docs]
``` -->
```bash
pip install git+ssh://github.com/cpviolation/analysis_helpers.git[docs]"
```

## Checking ROOT availability

ROOT (PyROOT) is an optional dependency. You can check whether it is available
before calling ROOT-dependent functions:

```python
from analysis_helpers import is_root_available

if is_root_available():
    # Safe to use ROOT-dependent helpers
    pass
else:
    print("PyROOT is not available in this environment")
```

To fail fast with a clear error message:

```python
from analysis_helpers import require_root

require_root()  # Raises ImportError when ROOT is missing
```

## Modules

| Module | Description |
|--------|-------------|
| [asymmetry](api/asymmetry.md) | Asymmetry computations |
| [dalitz](api/dalitz.md) | Dalitz plot utilities |
| [efficiency](api/efficiency.md) | Efficiency estimation |
| [fit](api/fit.md) | RooFit-based fitting helpers |
| [io](api/io.md) | Input-Output helpers |
| [kinematics](api/kinematics.md) | Kinematic calculations |
| [plotting](api/plotting.md) | Plotting utilities |
| [root_helpers](api/root_helpers.md) | PyROOT convenience wrappers |
| [utils](api/utils.md) | General utilities |
