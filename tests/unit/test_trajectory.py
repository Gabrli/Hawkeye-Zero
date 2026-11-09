import numpy as np
import pytest

from engine.trajectory import Trajectory


def test_predicts_constant_velocity_when_acceleration_is_zero():
    trajectory = Trajectory([0.0, 0.0], max_time=1.0)
    times, x, y = trajectory.predict_trajectory([2.0, 3.0], [4.0, -2.0])

    predicted = trajectory.predicted_position(times, x, y, 0.5)

    np.testing.assert_allclose(predicted, [4.0, 2.0], atol=1e-5)


def test_rejects_horizon_beyond_computed_curve():
    trajectory = Trajectory([0.0, 0.0], max_time=1.0)

    with pytest.raises(ValueError, match="longer than the ODE curve"):
        trajectory.predicted_position([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], 1.1)
