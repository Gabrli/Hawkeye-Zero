import numpy as np

from engine.history import ObservationHistory


def test_position_at_interpolates_between_observations():
    history = ObservationHistory()
    history.add(0.0, 0.0, 2.0)
    history.add(1.0, 10.0, 12.0)

    np.testing.assert_allclose(history.position_at(0.25), [2.5, 4.5])


def test_position_at_returns_none_outside_observed_range():
    history = ObservationHistory()
    history.add(1.0, 2.0, 3.0)
    history.add(2.0, 4.0, 5.0)

    assert history.position_at(0.5) is None
    assert history.position_at(2.5) is None


def test_reset_clears_recent_and_full_history():
    history = ObservationHistory()
    history.add(0.0, 0.0, 0.0)
    history.add(1.0, 1.0, 1.0)

    history.reset()

    assert len(history) == 0
    assert history.recent_points() == []
    assert history.recent_times() == []
    assert history.position_at(0.5) is None
