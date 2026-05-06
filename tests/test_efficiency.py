import numpy as np
import pytest

from analysis_helpers.efficiency import (
    binomial_error,
    get_efficiency,
    get_efficiency_array,
    get_sumw2,
    weighted_quantile,
    weighted_std,
    wilson_interval,
)


def test_get_sumw2_sums_squared_weights_per_bin():
    data = np.array([0.2, 0.3, 1.2, 1.8])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    bin_edges = np.array([0.0, 1.0, 2.0])

    result = get_sumw2(data, weights, bin_edges)

    assert np.allclose(result, np.array([5.0, 25.0]))


def test_weighted_quantile_matches_uniform_median():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones_like(values)

    result = weighted_quantile(values, weights, np.array([0.5]))

    assert result[0] == pytest.approx(2.5)


def test_weighted_quantile_returns_minimum_when_total_weight_non_positive():
    values = np.array([5.0, 2.0, 8.0])
    weights = np.zeros_like(values)

    result = weighted_quantile(values, weights, np.array([0.25, 0.75]))

    assert np.allclose(result, np.array([2.0, 2.0]))


def test_weighted_std_matches_manual_result():
    values = np.array([1.0, 2.0, 5.0])
    weights = np.array([1.0, 1.0, 2.0])

    result = weighted_std(values, weights)

    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    assert result == pytest.approx(np.sqrt(variance))


def test_weighted_std_rejects_non_positive_total_weight():
    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        weighted_std(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


def test_get_efficiency_array_without_weights_uses_binomial_error():
    passed = np.array([2.0, 8.0])
    total = np.array([4.0, 10.0])

    efficiency, interval = get_efficiency_array(passed, total)

    expected_efficiency = np.array([0.5, 0.8])
    expected_error = binomial_error(expected_efficiency, total)
    assert np.allclose(efficiency, expected_efficiency)
    assert np.allclose(interval[0], expected_efficiency - expected_error[0])
    assert np.allclose(interval[1], expected_efficiency + expected_error[1])


def test_get_efficiency_array_with_weights_uses_wilson_interval():
    passed = np.array([3.0])
    total = np.array([5.0])
    sumw2_total = np.array([5.0])

    efficiency, interval = get_efficiency_array(passed, total, sumw2_total=sumw2_total)

    expected_interval = np.array(wilson_interval(efficiency, total ** 2 / sumw2_total))
    assert efficiency[0] == pytest.approx(0.6)
    assert np.allclose(interval, expected_interval)


def test_get_efficiency_from_samples_matches_histogrammed_result():
    passed = np.array([0.2, 0.8, 1.2])
    total = np.array([0.2, 0.4, 0.8, 1.2, 1.8])
    bin_edges = np.array([0.0, 1.0, 2.0])

    efficiency, _ = get_efficiency(passed, total, bin_edges)

    assert np.allclose(efficiency, np.array([2.0 / 3.0, 1.0 / 2.0]))
