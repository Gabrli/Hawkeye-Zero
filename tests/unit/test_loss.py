import numpy as np

from engine.history import ObservationHistory
from engine.loss import PredictionLoss


def test_resolve_records_error_and_metrics():
    history = ObservationHistory()
    history.add(0.0, 0.0, 0.0)
    history.add(1.0, 3.0, 4.0)
    loss = PredictionLoss([1.0], history)
    loss.register(7, 0.0, 1.0, np.array([0.0, 0.0]))

    resolved = loss.resolve(1.0)

    assert len(resolved) == 1
    assert resolved[0][:4] == (7, 0.0, 1.0, 1.0)
    assert resolved[0][-1] == 5.0
    assert loss.mae(1.0) == 5.0
    assert loss.rmse(1.0) == 5.0


def test_reset_discards_pending_predictions_but_keeps_completed_errors():
    history = ObservationHistory()
    loss = PredictionLoss([1.0], history)
    loss.errors[1.0].append(2.0)
    loss.register(1, 0.0, 1.0, np.array([0.0, 0.0]))

    loss.reset()

    assert loss._pending == []
    assert loss.errors[1.0] == [2.0]
