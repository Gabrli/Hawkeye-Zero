import numpy as np

from engine.utils.calc_loss import calc_error


class PredictionLoss:
    """Evaluate queued predictions against later interpolated observations."""

    def __init__(self, horizons, history):
        self.horizons = horizons
        self.history = history

        self._pending = []
        self.errors = {h: [] for h in horizons}

    def register(self, frame_id, pred_time, horizon, predicted_xy, scale: float = 1.0):
        self._pending.append(
            (frame_id, pred_time, horizon, predicted_xy[0], predicted_xy[1], scale)
        )

    def reset(self):
        self._pending = []

    def resolve(self, current_time):
        remaining = []
        resolved = []

        for entry in self._pending:
            frame_id, pred_time, horizon, px, py, scale = entry
            target_time = pred_time + horizon

            if current_time >= target_time:
                real = self.history.position_at(target_time)

                if real is not None:
                    rx, ry = real
                    error_x = (px - rx) / scale
                    error_y = (py - ry) / scale
                    error = float(np.hypot(error_x, error_y))

                    self.errors[horizon].append(error)

                    resolved.append(
                        (
                            frame_id,
                            pred_time,
                            horizon,
                            target_time,
                            px,
                            py,
                            rx,
                            ry,
                            error_x,
                            error_y,
                            error,
                        )
                    )
            else:
                remaining.append(entry)

        self._pending = remaining
        return resolved

    def mae(self, horizon):
        errors = self.errors[horizon]
        return calc_error(errors, "mae") if errors else None

    def rmse(self, horizon):
        errors = self.errors[horizon]
        return calc_error(errors, "rmse") if errors else None
