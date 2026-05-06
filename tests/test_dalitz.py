import numpy as np
import pytest

from analysis_helpers.dalitz import (
    DalitzKinematics,
    cos_helicity_angle_dalitz,
    two_body_momentum,
    two_body_momentum_squared,
    vector,
)


def test_two_body_momentum_squared_matches_momentum_squared():
    momentum_squared = two_body_momentum_squared(5.0, 1.0, 2.0)
    momentum = two_body_momentum(5.0, 1.0, 2.0)

    assert momentum ** 2 == pytest.approx(momentum_squared)


def test_vector_stacks_components_on_last_axis():
    result = vector(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]))

    assert result.shape == (2, 3)
    assert np.allclose(result[0], np.array([1.0, 3.0, 5.0]))


def test_cos_helicity_angle_is_bounded_for_physical_point():
    result = cos_helicity_angle_dalitz(4.0, 4.5, 3.0, 0.5, 0.6, 0.7)

    assert -1.0 <= result <= 1.0


def test_dalitz_kinematics_basic_methods_return_expected_values():
    dalitz = DalitzKinematics(5.0, [1.0, 1.5, 2.0])

    assert dalitz.otherDaughter([0, 2]) == 1
    assert dalitz.mSqMin([0, 1]) == pytest.approx((1.0 + 1.5) ** 2)
    assert dalitz.mSqMax([0, 1]) == pytest.approx((5.0 - 2.0) ** 2)
    assert dalitz.EStar2(3.0, [0, 1]) == pytest.approx((3.0 ** 2 - 1.0 ** 2 + 1.5 ** 2) / (2.0 * 3.0))
    assert dalitz.EStar3(3.0, [0, 1]) == pytest.approx((5.0 ** 2 - 3.0 ** 2 - 2.0 ** 2) / (2.0 * 3.0))
    assert dalitz.OtherMassSq(4.0, 5.0) == pytest.approx(5.0 ** 2 - 4.0 - 5.0 + 1.0 ** 2 + 1.5 ** 2 + 2.0 ** 2)


def test_dalitz_kinematics_contour_shape_and_bounds():
    dalitz = DalitzKinematics(5.0, [1.0, 1.5, 2.0])

    m_sq, y_min, y_max = dalitz.Contour([0, 1], num_points=25)

    assert len(m_sq) == 25
    assert len(y_min) == 25
    assert len(y_max) == 25
    assert np.all(y_min <= y_max)


@pytest.mark.parametrize("bad_indices", [[], [0], [0, 1, 2]])
def test_dalitz_kinematics_rejects_invalid_daughter_index_lists(bad_indices):
    dalitz = DalitzKinematics(5.0, [1.0, 1.5, 2.0])

    with pytest.raises(ValueError, match="Two daughters"):
        dalitz.otherDaughter(bad_indices)


def test_dalitz_kinematics_requires_three_daughters():
    with pytest.raises(ValueError, match="Three daughters"):
        DalitzKinematics(5.0, [1.0, 2.0])