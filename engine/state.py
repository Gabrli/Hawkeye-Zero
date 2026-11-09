import numpy as np

MAX_ACCEL_MAGNITUDE = 150.0
VELOCITY_SMOOTHING = 0.6
ACCEL_SMOOTHING = 0.08


class StateEstimator:
    """Estimate smoothed position, velocity, and acceleration from observations."""

    def __init__(
        self,
        max_accel_magnitude: float = MAX_ACCEL_MAGNITUDE,
        velocity_smoothing: float = VELOCITY_SMOOTHING,
        accel_smoothing: float = ACCEL_SMOOTHING,
    ):
        """Configure acceleration limits and exponential smoothing factors.

        Args:
            max_accel_magnitude: Maximum magnitude allowed for fitted acceleration.
            velocity_smoothing: Weight of the newest velocity estimate.
            accel_smoothing: Weight of the newest acceleration estimate.
        """
        self.max_accel_magnitude = max_accel_magnitude
        self.velocity_smoothing = velocity_smoothing
        self.accel_smoothing = accel_smoothing

        self._smoothed_velocity = None
        self._smoothed_acceleration = None

    def reset(self):
        """Clear smoothed velocity and acceleration estimates."""
        self._smoothed_velocity = None
        self._smoothed_acceleration = None

    def estimate(self, points, times):
        """Estimate the current position and smoothed motion state.

        Args:
            points: Recent ``(x, y)`` observations in chronological order.
            times: Observation times in seconds, corresponding to ``points``.

        Returns:
            A ``(position, velocity, acceleration)`` tuple of two-element NumPy
            arrays, or ``None`` if the quadratic fit cannot be computed.
        """
        raw = self._fit(points, times)

        if raw is None:
            return None

        position, velocity, acceleration = raw

        if self._smoothed_velocity is None:
            self._smoothed_velocity = velocity
        else:
            self._smoothed_velocity = (
                self.velocity_smoothing * velocity
                + (1 - self.velocity_smoothing) * self._smoothed_velocity
            )

        if self._smoothed_acceleration is None:
            self._smoothed_acceleration = acceleration
        else:
            self._smoothed_acceleration = (
                self.accel_smoothing * acceleration
                + (1 - self.accel_smoothing) * self._smoothed_acceleration
            )

        return position, self._smoothed_velocity, self._smoothed_acceleration

    def _fit(self, points, times):
        """Fit independent quadratic polynomials to the x and y observations.

        Args:
            points: Position samples with shape ``(n, 2)``.
            times: Sample times with shape ``(n,)``.

        Returns:
            Position, velocity, and acceleration arrays evaluated at the latest
            sample, or ``None`` when fewer than five samples are available or
            the polynomial fit is singular.
        """
        points = np.asarray(points, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        if len(points) < 5:
            return None

        t = times - times[-1]

        try:
            px = np.polyfit(t, points[:, 0], 2)
            py = np.polyfit(t, points[:, 1], 2)
        except np.linalg.LinAlgError:
            return None

        x, y = px[2], py[2]
        vx, vy = px[1], py[1]
        ax, ay = 2 * px[0], 2 * py[0]

        accel_magnitude = np.hypot(ax, ay)
        if accel_magnitude > self.max_accel_magnitude:
            scale = self.max_accel_magnitude / accel_magnitude
            ax *= scale
            ay *= scale

        return np.array([x, y]), np.array([vx, vy]), np.array([ax, ay])