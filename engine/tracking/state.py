import numpy as np

MAX_ACCEL_MAGNITUDE = 150.0
VELOCITY_SMOOTHING = 0.6
ACCEL_SMOOTHING = 0.08


class StateEstimator:
    """Estimate 2D position, velocity, and acceleration from recent samples."""

    def __init__(
        self,
        max_accel_magnitude: float = MAX_ACCEL_MAGNITUDE,
        velocity_smoothing: float = VELOCITY_SMOOTHING,
        accel_smoothing: float = ACCEL_SMOOTHING,
    ):
        self.max_accel_magnitude = max_accel_magnitude
        self.velocity_smoothing = velocity_smoothing
        self.accel_smoothing = accel_smoothing

        self._smoothed_velocity = None
        self._smoothed_acceleration = None

    def reset(self):
        self._smoothed_velocity = None
        self._smoothed_acceleration = None

    def estimate(self, points, times):
        """Return the latest position, smoothed velocity, and smoothed acceleration."""
        fitted_state = self._fit(points, times)

        if fitted_state is None:
            return None

        position, velocity, acceleration = fitted_state

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
        points = np.asarray(points, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 2 or times.ndim != 1:
            raise ValueError("points must have shape (N, 2) and times shape (N,)")
        if len(points) != len(times):
            raise ValueError("points and times must contain the same number of samples")
        if not np.isfinite(points).all() or not np.isfinite(times).all():
            raise ValueError("points and times must contain only finite values")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("observation times must be strictly increasing")
        if len(points) < 5:
            return None

        t = times - times[-1]

        try:
            x_coefficients = np.polyfit(t, points[:, 0], 2)
            y_coefficients = np.polyfit(t, points[:, 1], 2)
        except np.linalg.LinAlgError:
            return None

        x, y = x_coefficients[2], y_coefficients[2]
        vx, vy = x_coefficients[1], y_coefficients[1]
        ax, ay = 2 * x_coefficients[0], 2 * y_coefficients[0]

        accel_magnitude = np.hypot(ax, ay)
        if accel_magnitude > self.max_accel_magnitude:
            scale = self.max_accel_magnitude / accel_magnitude
            ax *= scale
            ay *= scale

        return np.array([x, y]), np.array([vx, vy]), np.array([ax, ay])
