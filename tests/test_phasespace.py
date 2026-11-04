import numpy as np
import pytest

from analysis_helpers.phasespace import (
    CombinedPhaseSpace,
    DalitzPhaseSpace,
    DecayTimePhaseSpace,
)


@pytest.fixture
def dalitz_phase_space():
    return DalitzPhaseSpace(0.1, 0.1, 0.1, 1.0)


def test_dalitz_phase_space_geometry_and_jacobians(dalitz_phase_space):
    sample = np.array([[0.25, 0.25]])

    assert dalitz_phase_space.dimensionality() == 2
    assert np.allclose(
        dalitz_phase_space.bounds(),
        [(0.04, 0.81), (0.04, 0.81)],
    )
    assert dalitz_phase_space.inside(sample).tolist() == [True]
    assert dalitz_phase_space.m2ac(sample).tolist() == pytest.approx([0.53])
    assert dalitz_phase_space.invariant_mass_jacobian(sample).tolist() == pytest.approx(
        [1.0]
    )
    assert np.isfinite(dalitz_phase_space.square_dalitz_plot_jacobian(sample)).all()


@pytest.mark.parametrize("space_to_sample", ["DP", "linDP", "sqDP"])
def test_dalitz_rectangular_grid_sample_handles_unequal_dimensions(
    dalitz_phase_space, space_to_sample
):
    sample = dalitz_phase_space.rectangular_grid_sample(
        7, 5, space_to_sample=space_to_sample
    )

    assert sample.ndim == 2
    assert sample.shape[1] == 2
    assert len(sample) > 0
    assert dalitz_phase_space.inside(sample).all()


def test_decay_time_samples_are_positive_and_include_optional_majorant():
    phase_space = DecayTimePhaseSpace(tau=2.0)
    sample = phase_space.unfiltered_sample(100, maximum=3.0)

    assert phase_space.dimensionality() == 1
    assert phase_space.bounds() == [(0.0, float("inf"))]
    assert sample.shape == (100, 2)
    assert (sample[:, 0] > 0).all()
    assert ((sample[:, 1] >= 0) & (sample[:, 1] <= 3.0)).all()
    assert np.array_equal(phase_space.t(sample), sample[:, 0])
    assert phase_space.inside(sample).all()


def test_combined_phase_space_concatenates_and_filters_samples(dalitz_phase_space):
    phase_space = CombinedPhaseSpace(
        dalitz_phase_space, DecayTimePhaseSpace(tau=1.5)
    )
    sample = phase_space.unfiltered_sample(500, maximum=2.0)

    assert phase_space.dimensionality() == 3
    assert phase_space.bounds() == dalitz_phase_space.bounds() + [
        (0.0, float("inf"))
    ]
    assert sample.shape == (500, 4)
    assert phase_space.data1(sample).shape == (500, 2)
    assert phase_space.data2(sample).shape == (500, 2)

    filtered = phase_space.filter(sample)

    assert len(filtered) > 0
    assert phase_space.inside(filtered).all()


def test_decay_time_truncated_sampling_respects_window_without_rejection():
    phase_space = DecayTimePhaseSpace(tau=1.0, tmin=2.0, tmax=5.0)
    sample = phase_space.unfiltered_sample(5000)

    assert phase_space.bounds() == [(2.0, 5.0)]
    assert len(phase_space.filter(sample)) == 5000
    assert ((sample > 2.0) & (sample < 5.0)).all()


def test_decay_time_tmin_only_is_shifted_exponential():
    phase_space = DecayTimePhaseSpace(tau=1.5, tmin=3.0)
    t = phase_space.t(phase_space.unfiltered_sample(50000))

    assert (t > 3.0).all()
    assert np.mean(t - 3.0) == pytest.approx(1.5, rel=0.05)


def test_decay_time_rejects_invalid_window():
    with pytest.raises(ValueError):
        DecayTimePhaseSpace(tmin=2.0, tmax=1.0)
