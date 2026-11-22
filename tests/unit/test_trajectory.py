import numpy as np
import pytest

from engine.trajectory import Trajectory


def test_predicts_constant_velocity_when_acceleration_is_zero():
    trajectory = Trajectory([0.0, 0.0], max_time=1.0)
    times, x_positions, y_positions = trajectory.predict_trajectory(
        [2.0, 3.0], [4.0, -2.0]
    )

    predicted = trajectory.predicted_position(times, x_positions, y_positions, 0.5)

    np.testing.assert_allclose(predicted, [4.0, 2.0], atol=1e-5)


def test_rejects_horizon_beyond_computed_curve():
    trajectory = Trajectory([0.0, 0.0], max_time=1.0)

    with pytest.raises(ValueError, match="longer than the ODE curve"):
        trajectory.predicted_position([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], 1.1)


def test_acceleration_decays_over_time():
    trajectory = Trajectory([10.0, -5.0], max_time=1.0)
    initial_derivative = trajectory.ode(0.0, [0.0, 0.0, 0.0, 0.0])
    later_derivative = trajectory.ode(0.4, [0.0, 0.0, 0.0, 0.0])

    assert abs(later_derivative[2]) < abs(initial_derivative[2])
    assert abs(later_derivative[3]) < abs(initial_derivative[3])


def test_rejects_invalid_solver_method():
    trajectory = Trajectory([0.0, 0.0], runge_kutta_method="not-a-solver", max_time=1.0)

    with pytest.raises((ValueError, TypeError)):
        trajectory.predict_trajectory([0.0, 0.0], [1.0, 1.0])
