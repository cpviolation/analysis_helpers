import pytest
import numpy as np

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


class _FakeObs:
    def __init__(self, name, bins, xmin, xmax):
        self.name = name
        self._bins = bins
        self._xmin = xmin
        self._xmax = xmax

    def getBins(self):
        return self._bins

    def getMin(self):
        return self._xmin

    def getMax(self):
        return self._xmax

    def InheritsFrom(self, cls_name):
        return cls_name == "RooAbsArg"


class _FakeRooArgSet:
    def __init__(self):
        self.items = []

    def add(self, obj):
        self.items.append(obj)


def _fake_root_backend(captured):
    class _FakeRooDataHist:
        @staticmethod
        def from_numpy(nphist, observables, bins=None, ranges=None):
            captured["nphist_shape"] = tuple(nphist.shape)
            captured["nphist_sum"] = int(nphist.sum())
            captured["observables"] = observables
            captured["bins"] = bins
            captured["ranges"] = ranges
            return {"ok": True, "bins": bins, "ranges": ranges}

    class _FakeROOT:
        RooArgSet = _FakeRooArgSet
        RooDataHist = _FakeRooDataHist

    return _FakeROOT()


def test_datahist_from_numpy_supports_2d_arrays(monkeypatch):
    captured = {}
    monkeypatch.setattr(fit, "r", _fake_root_backend(captured))

    fu = fit.FitUtils(_name="unit-test")
    x = _FakeObs("x", bins=10, xmin=-1.0, xmax=1.0)
    y = _FakeObs("y", bins=20, xmin=0.0, xmax=5.0)
    arr = np.array([
        [-0.5, 1.0],
        [0.2, 2.1],
        [0.0, 4.2],
        [0.7, 3.3],
    ])

    out = fu.DataHistFromNumpy(arr, [x, y])

    assert out["ok"] is True
    assert captured["bins"] == [10, 20]
    assert captured["ranges"] == [(-1.0, 1.0), (0.0, 5.0)]
    assert captured["nphist_shape"] == (10, 20)
    assert len(captured["observables"].items) == 2


def test_datahist_from_numpy_accepts_transposed_2d_input(monkeypatch):
    captured = {}
    monkeypatch.setattr(fit, "r", _fake_root_backend(captured))

    fu = fit.FitUtils(_name="unit-test")
    x = _FakeObs("x", bins=8, xmin=-2.0, xmax=2.0)
    y = _FakeObs("y", bins=6, xmin=-3.0, xmax=3.0)

    # Shape (n_observables, n_samples): should be auto-transposed.
    arr_t = np.array([
        [-1.0, 0.0, 0.5, 1.0],
        [-2.0, -1.0, 1.0, 2.0],
    ])

    fu.DataHistFromNumpy(arr_t, [x, y])

    assert captured["nphist_shape"] == (8, 6)
    assert captured["nphist_sum"] == 4


def test_datahist_from_numpy_rejects_shape_observable_mismatch(monkeypatch):
    captured = {}
    monkeypatch.setattr(fit, "r", _fake_root_backend(captured))

    fu = fit.FitUtils(_name="unit-test")
    x = _FakeObs("x", bins=10, xmin=0.0, xmax=1.0)
    y = _FakeObs("y", bins=10, xmin=0.0, xmax=1.0)
    z = _FakeObs("z", bins=10, xmin=0.0, xmax=1.0)

    arr = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    with pytest.raises(ValueError, match="incompatible with 3 observables"):
        fu.DataHistFromNumpy(arr, [x, y, z])
