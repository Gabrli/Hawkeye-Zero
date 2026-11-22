import numpy as np


HISTORY_LENGTH = 1000
HISTORY_SECONDS = 2.0


class ObservationHistory:
    """Store recent observations and retain a full timeline for interpolation."""

    def __init__(
        self,
        history_seconds: float = HISTORY_SECONDS,
        history_length: int = HISTORY_LENGTH,
    ):
        self.history_seconds = history_seconds
        self.history_length = history_length

        self._recent_points = []
        self._recent_times = []

        self._all_times = []
        self._all_x = []
        self._all_y = []

    def add(self, time: float, x: float, y: float):
        """Append one observation. Timestamps must be strictly increasing."""
        if not np.isfinite([time, x, y]).all():
            raise ValueError("observation time and coordinates must be finite")
        if self._all_times and time <= self._all_times[-1]:
            raise ValueError("observation times must be strictly increasing")
        self._recent_points.append([x, y])
        self._recent_times.append(time)

        while (
            len(self._recent_times) > 2
            and (self._recent_times[-1] - self._recent_times[0]) > self.history_seconds
        ):
            self._recent_points.pop(0)
            self._recent_times.pop(0)

        if len(self._recent_points) > self.history_length:
            self._recent_points.pop(0)
            self._recent_times.pop(0)

        self._all_times.append(time)
        self._all_x.append(x)
        self._all_y.append(y)

    def reset(self):
        """Clear the history of observations."""
        self._recent_points.clear()
        self._recent_times.clear()
        self._all_times.clear()
        self._all_x.clear()
        self._all_y.clear()

    def recent_points(self):
        return self._recent_points

    def recent_times(self):
        return self._recent_times

    def __len__(self):
        return len(self._recent_points)

    def position_at(self, target_time: float):
        """Interpolate an observed position, or return None outside the history."""
        if len(self._all_times) < 2:
            return None

        times = np.asarray(self._all_times)

        if target_time < times[0] or target_time > times[-1]:
            return None

        xs = np.asarray(self._all_x)
        ys = np.asarray(self._all_y)

        x = np.interp(target_time, times, xs)
        y = np.interp(target_time, times, ys)

        return np.array([x, y])
