"""Shared helpers to import PyROOT with a lazy error proxy fallback."""


def is_root_available(has_root):
    """Return whether PyROOT is available for the calling module."""
    return has_root


def require_root(has_root, error_message):
    """Raise ImportError with a clear message if PyROOT is not available."""
    if not has_root:
        raise ImportError(error_message)


def import_root_with_proxy(error_message="PyROOT is not available. Install and configure ROOT to use ROOT-dependent fit tools."):
    """Return (ROOT module or proxy, availability flag).

    The proxy defers the ImportError until a ROOT attribute is accessed.
    """
    try:
        import ROOT as r  # type: ignore[import-not-found]

        return r, True
    except ModuleNotFoundError:

        class _MissingROOT:
            """Proxy that raises a clear error only when ROOT-backed code is used."""

            def __getattr__(self, _name):
                raise ImportError(error_message)

        return _MissingROOT(), False
