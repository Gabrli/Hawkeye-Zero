import numpy as np

from engine.foe.estimator import DescentEstimator, trajectory_points


def test_estimates_focus_of_expansion_and_time_to_contact():
    fps = 20.0
    scale = 1.1
    expected_foe = np.array([120.0, 80.0])
    linear_transform = np.eye(2) * scale
    translation = (1.0 - scale) * expected_foe
    frame_step = np.eye(3)
    frame_step[:2, :2] = linear_transform
    frame_step[:2, 2] = translation

    estimator = DescentEstimator(fps=fps, min_steps=5)
    estimate = None
    for _ in range(5):
        estimate = estimator.update(frame_step, reliable=True)

    assert estimate is not None
    estimated_foe, estimated_time = estimate
    np.testing.assert_allclose(estimated_foe, expected_foe, atol=1e-6)
    expected_time = (1.0 / fps) / (1.0 - 1.0 / scale)
    assert abs(estimated_time - expected_time) < 1e-9


def test_trajectory_points_omit_times_after_contact():
    points, landing = trajectory_points(
        camera_center=(0, 0), focus_of_expansion=(100, 50), time_to_contact=2.0, ticks=(0.5, 1.0, 2.0, 3.0)
    )

    assert [time for time, _ in points] == [0.5, 1.0]
    assert landing == (2.0, (100.0, 50.0))
