import numpy as np
import pytest

from analysis_helpers.asymmetry import get_asymmetry_array


def test_get_asymmetry_array_matches_expected_values():
    arr1 = np.array([9.0, 4.0])
    arr2 = np.array([1.0, 4.0])

    asymmetry, asymmetry_err = get_asymmetry_array(arr1, arr2)

    expected_asymmetry = np.array([0.8, 0.0])
    expected_error = np.array([
        np.sqrt(4.0 * (arr2[0] ** 2 * arr1[0] + arr1[0] ** 2 * arr2[0]) / (arr1[0] + arr2[0]) ** 4),
        np.sqrt(4.0 * (arr2[1] ** 2 * arr1[1] + arr1[1] ** 2 * arr2[1]) / (arr1[1] + arr2[1]) ** 4),
    ])

    assert np.allclose(asymmetry, expected_asymmetry)
    assert np.allclose(asymmetry_err, expected_error)


def test_get_asymmetry_array_uses_explicit_sumw2_and_correlation():
    arr1 = np.array([10.0])
    arr2 = np.array([6.0])
    sumw2_arr1 = np.array([4.0])
    sumw2_arr2 = np.array([9.0])
    correlation = 0.25

    asymmetry, asymmetry_err = get_asymmetry_array(
        arr1,
        arr2,
        sumw2_arr1=sumw2_arr1,
        sumw2_arr2=sumw2_arr2,
        correlation=correlation,
    )

    numerator = arr1[0] - arr2[0]
    denominator = arr1[0] + arr2[0]
    expected_err2 = 4.0 * (
        arr2[0] ** 2 * sumw2_arr1[0]
        + arr1[0] ** 2 * sumw2_arr2[0]
        - 2.0 * correlation * arr1[0] * arr2[0] * np.sqrt(sumw2_arr1[0]) * np.sqrt(sumw2_arr2[0])
    ) / denominator ** 4

    assert asymmetry[0] == pytest.approx(numerator / denominator)
    assert asymmetry_err[0] == pytest.approx(np.sqrt(expected_err2))
