import numpy as np
import pytest
import scipy.constants
import vector

from analysis_helpers.kinematics import (
    ctau,
    ctau_from_tau,
    dira,
    distance,
    doca,
    estar2,
    estar3,
    ip_all,
    mass,
    momentum,
    opening_angle,
    pseudorapidity,
    pt,
    slope,
    phi,
)


def _as_scalar(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise TypeError(f"Expected a scalar or one-element array, got shape {array.shape}")


def test_ip_all_returns_distance_from_point_to_line():
    result = ip_all(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 10.0)
    assert result == pytest.approx(5.0)


def test_doca_returns_zero_for_intersecting_lines():
    result = doca(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0)
    assert result == pytest.approx(0.0)


def test_dira_is_one_for_aligned_momentum_and_flight_direction():
    result = dira(2.0, 0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert result == pytest.approx(1.0)


def test_opening_angle_matches_known_geometry():
    result = opening_angle(1.0, 0.0, 1.0, 0.0, 1.0, 1.0)
    assert result == pytest.approx(np.pi / 3.0)


def test_scalar_helpers_return_expected_values():
    assert ctau(6.0, 2.0, 4.0) == pytest.approx(3.0)
    assert ctau_from_tau(1.0) == pytest.approx(scipy.constants.c / 1e6)
    assert distance(0.0, 0.0, 0.0, 3.0, 4.0, 12.0) == pytest.approx(13.0)
    assert slope(5.0, 12.0) == pytest.approx(5.0 / 12.0)


def test_vector_based_kinematics_match_expected_values():
    px, py, pz, energy = 3.0, 4.0, 12.0, 14.0
    reference = vector.obj(px=px, py=py, pz=pz, E=energy)

    assert _as_scalar(mass(px, py, pz, energy)) == pytest.approx(
        _as_scalar(reference.mass))
    assert _as_scalar(momentum(px, py, pz)) == pytest.approx(13.0)
    assert _as_scalar(pt(px, py)) == pytest.approx(5.0)
    assert _as_scalar(pseudorapidity(px, py, pz)) == pytest.approx(_as_scalar(reference.eta))
    assert _as_scalar(phi(px, py, pz)) == pytest.approx(_as_scalar(reference.phi))


def test_estar_helpers_compute_expected_values():
    masses = np.array([1.0, 2.0, 3.0, 4.0])

    assert estar2(5.0, masses) == pytest.approx(3.0)
    assert estar3(5.0, masses) == pytest.approx(-4.0)


@pytest.mark.parametrize(
    "bad_masses",
    [
        [1.0, 2.0, 3.0],
        np.array([[1.0, 2.0, 3.0, 4.0]]),
    ],
)
def test_estar_helpers_reject_non_four_element_mass_inputs(bad_masses):
    with pytest.raises(ValueError, match="exactly 4 elements"):
        estar2(5.0, bad_masses)

    with pytest.raises(ValueError, match="exactly 4 elements"):
        estar3(5.0, bad_masses)
