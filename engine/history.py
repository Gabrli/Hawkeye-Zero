import numpy as np


HISTORY_LENGTH = 1000
HISTORY_SECONDS = 2.0


class ObservationHistory:
    """Store recent observations and interpolate positions across video time."""

    def __init__(self, history_seconds: float = HISTORY_SECONDS, history_length: int = HISTORY_LENGTH):
        """Create bounded recent history and an unbounded interpolation history.

        Args:
            history_seconds: Maximum time span retained in the recent window.
            history_length: Maximum number of observations retained in the window.
        """
        self.history_seconds = history_seconds
        self.history_length = history_length

        self._recent_points = []
        self._recent_times = []

        self._all_times = []
        self._all_x = []
        self._all_y = []

    def add(self, time: float, x: float, y: float):
        """Add a timestamped position observation.

        Args:
            time: Observation time in seconds.
            x: Horizontal position in the active coordinate system.
            y: Vertical position in the active coordinate system.
        """
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
        """Remove all recent and interpolation observations."""
        self._recent_points.clear()
        self._recent_times.clear()
        self._all_times.clear()
        self._all_x.clear()
        self._all_y.clear()

    def recent_points(self):
        """Return positions in the bounded recent window.

        Returns:
            The stored ``[x, y]`` point list in chronological order.
        """
        return self._recent_points

    def recent_times(self):
        """Return timestamps corresponding to :meth:`recent_points`.

        Returns:
            The stored timestamp list in chronological order.
        """
        return self._recent_times

    def __len__(self):
        """Return the number of observations in the recent window."""
        return len(self._recent_points)

    def position_at(self, target_time: float):
        """Interpolate the observed position at a time within the video history.

        Args:
            target_time: Time in seconds at which to estimate a position.

        Returns:
            A NumPy ``[x, y]`` array, or ``None`` if fewer than two observations
            exist or the time is outside the observed interval.
        """
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