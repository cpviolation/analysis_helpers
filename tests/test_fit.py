import pytest

import analysis_helpers.fit as fit


def test_fit_module_exposes_root_availability_flag():
    assert isinstance(fit._HAS_ROOT, bool)


def test_fit_models_constructor_is_import_safe():
    model = fit.FitModels(_name="unit-test")
    assert model.name == "unit-test"


def test_ws_fails_cleanly_without_root():
    if fit._HAS_ROOT:
        pytest.skip("PyROOT is available; optional-ROOT fallback path is not active")

    with pytest.raises(ImportError, match="PyROOT is not available"):
        fit.WS(None, None)
