import numpy as np

from engine.state import StateEstimator


def test_estimate_fits_position_and_velocity():
    times = np.arange(5, dtype=float) * 0.1
    points = [[10.0 + 2.0 * time, 5.0 - 3.0 * time] for time in times]

    position, velocity, acceleration = StateEstimator().estimate(points, times)

    np.testing.assert_allclose(position, [10.8, 3.8], atol=1e-10)
    np.testing.assert_allclose(velocity, [2.0, -3.0], atol=1e-10)
    np.testing.assert_allclose(acceleration, [0.0, 0.0], atol=1e-10)


def test_reset_discards_smoothed_state():
    estimator = StateEstimator()
    estimator.estimate([[float(i), 0.0] for i in range(5)], [float(i) for i in range(5)])

    estimator.reset()

    assert estimator._smoothed_velocity is None
    assert estimator._smoothed_acceleration is None
