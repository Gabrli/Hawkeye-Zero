import numpy as np

from engine.utils import calc_error


class PredictionLoss:
    """Compare registered future predictions with observed positions."""

    def __init__(self, horizons, history):
        """Create a prediction evaluator attached to an observation history.

        Args:
            horizons: Prediction horizons in seconds to track.
            history: Observation history used to retrieve later ground truth.
        """
        self.horizons = horizons
        self.history = history

        self._pending = []
        self.errors = {h: [] for h in horizons}

    def reset(self):
        """Discard predictions that have not yet been resolved."""
        self._pending.clear()

    def register(self, frame_id, pred_time, horizon, predicted_xy):
        """Queue a prediction for comparison when its target time is observed.

        Args:
            frame_id: Source frame index of the prediction.
            pred_time: Time at which the prediction was made, in seconds.
            horizon: Future prediction horizon in seconds.
            predicted_xy: Predicted ``(x, y)`` position.
        """
        self._pending.append((frame_id, pred_time, horizon, predicted_xy[0], predicted_xy[1]))

    def resolve(self, current_time):
        """Resolve due predictions against interpolated observations.

        Args:
            current_time: Latest processed video time in seconds.

        Returns:
            Rows for predictions with available ground truth. Each row contains
            frame index, prediction and target times, horizon, predicted and
            actual positions, component errors, and Euclidean error.
        """
        remaining = []
        resolved = []

        for entry in self._pending:
            frame_id, pred_time, horizon, px, py = entry
            target_time = pred_time + horizon

            if current_time >= target_time:
                real = self.history.position_at(target_time)

                if real is not None:
                    rx, ry = real
                    error_x = px - rx
                    error_y = py - ry
                    error = float(np.hypot(error_x, error_y))

                    self.errors[horizon].append(error)

                    resolved.append(
                        (frame_id, pred_time, horizon, target_time, px, py, rx, ry, error_x, error_y, error)
                    )
            else:
                remaining.append(entry)

        self._pending = remaining
        return resolved

    def mae(self, horizon):
        """Return mean absolute Euclidean error for a prediction horizon.

        Args:
            horizon: Prediction horizon in seconds.

        Returns:
            Mean error in pixels, or ``None`` if no errors have been recorded.
        """
        errors = self.errors[horizon]
        return calc_error(errors, "mae") if errors else None

    def rmse(self, horizon):
        """Return root mean square Euclidean error for a prediction horizon.

        Args:
            horizon: Prediction horizon in seconds.

        Returns:
            Root mean square error in pixels, or ``None`` if no errors have
            been recorded.
        """
        errors = self.errors[horizon]
        return calc_error(errors, "rmse") if errors else None