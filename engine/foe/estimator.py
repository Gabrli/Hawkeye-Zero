import numpy as np
from engine.foe import WINDOW_S, SMOOTH, MIN_STEPS, DT, FPS, TICKS

class DescentEstimator:
    """Estimate the image focus of expansion and approximate time-to-contact.

    Input transforms must map points from the previous frame to the current
    frame. Only reliable transforms with apparent expansion are retained.
    """

    def __init__(
        self,
        fps: float = FPS,
        window_s: float = WINDOW_S,
        smooth: float = SMOOTH,
        min_steps: int = MIN_STEPS,
    ):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        if not 0.0 <= smooth <= 1.0:
            raise ValueError("smooth must be between 0 and 1")
        if window_s <= 0 or min_steps < 1:
            raise ValueError("window_s and min_steps must be positive")
        self.dt = 1.0 / float(fps)
        self.window = max(int(round(window_s * fps)), min_steps)
        self.smoothing = smooth
        self.min_steps = min_steps
        self._recent_steps = []  # (linear transform, translation, scale)
        self._smoothed_foe = None
        self._smoothed_time_to_contact = None

    def reset(self):
        self._recent_steps.clear()
        self._smoothed_foe = self._smoothed_time_to_contact = None

    def update(self, step, reliable: bool):
        """step: 3x3 prev -> pcur (homogeneous coordinates)."""
        if reliable and step is not None:
            linear_transform = step[:2, :2]
            translation = step[:2, 2]
            scale = float(np.sqrt(abs(np.linalg.det(linear_transform))))
            if scale > 1.0:
                self._recent_steps.append((linear_transform.copy(), translation.copy(), scale))
        self._recent_steps = self._recent_steps[-self.window :]
        estimate = self._estimate()
        if estimate is None:
            if self._smoothed_foe is None:
                return None
            return self._smoothed_foe, self._smoothed_time_to_contact

        focus_of_expansion, time_to_contact = estimate
        if self._smoothed_foe is None:
            self._smoothed_foe = focus_of_expansion
            self._smoothed_time_to_contact = time_to_contact
        else:
            weight = self.smoothing
            self._smoothed_foe = (
                (1 - weight) * self._smoothed_foe + weight * focus_of_expansion
            )
            self._smoothed_time_to_contact = (
                (1 - weight) * self._smoothed_time_to_contact
                + weight * time_to_contact
            )
        return self._smoothed_foe, self._smoothed_time_to_contact

    def _estimate(self):
        # estimate the FOE (focus of expansion) -> the point (x, y) in the image where the camera is moving towards, and tau -> time to landing in seconds, based on the recent reliable steps
        if len(self._recent_steps) < self.min_steps:
            return None
        identity = np.eye(2)
        linear_systems = [identity - transform for transform, _, _ in self._recent_steps]
        translations = [offset for _, offset, _ in self._recent_steps]
        inlier_mask = np.ones(len(linear_systems), dtype=bool)
        focus_of_expansion = None
        for _ in range(3):
            # compare data from the last few steps to estimate the FOE and filter out outliers based on the median absolute deviation
            # (LS) -> method of least squares to solve for the FOE (focus of expansion) given the recent steps
            normal_matrix = sum(
                matrix.T @ matrix
                for matrix, keep in zip(linear_systems, inlier_mask)
                if keep
            )
            right_hand_side = sum(
                matrix.T @ offset
                for matrix, offset, keep in zip(linear_systems, translations, inlier_mask)
                if keep
            )
            if abs(np.linalg.det(normal_matrix)) < 1e-12:
                return None
            focus_of_expansion = np.linalg.solve(normal_matrix, right_hand_side)
            residuals = np.array([
                np.linalg.norm(matrix @ focus_of_expansion - offset)
                for matrix, offset in zip(linear_systems, translations)
            ])
            median_residual = np.median(residuals)
            median_absolute_deviation = np.median(np.abs(residuals - median_residual)) + 1e-9
            inlier_mask = residuals < median_residual + 3 * 1.4826 * median_absolute_deviation
            if inlier_mask.sum() < self.min_steps:
                return None
        # MAD (median absolute deviation) -> a robust measure of the variability of the residuals, used to filter out outliers
        median_scale = float(np.median([scale for (_, _, scale), keep in zip(self._recent_steps, inlier_mask) if keep]))
        if median_scale <= 1.0 + 1e-6:
            return None
        # estimate the time to landing (tau) based on the scale of the camera motion
        # ta = dt / (1 - 1/m) -> time to landing in seconds, where dt is the time between frames and m is the scale of the camera motion
        time_to_contact = self.dt / (1.0 - 1.0 / median_scale)
        # estimated position p(h) = c + (f - c) * (h / tau), where c is the camera center, f is the FOE, h is the time to landing, and tau is the time to landing
        return focus_of_expansion, time_to_contact


def trajectory_points(camera_center, focus_of_expansion, time_to_contact, ticks=TICKS):
    """Return image-space points along the estimated path to the FOE.

    Args:
        camera_center: Camera center as an ``(x, y)`` pair in pixels.
        focus_of_expansion: Estimated FOE as an ``(x, y)`` pair in pixels.
        time_to_contact: Estimated time to contact, in seconds.
        ticks: Requested times in seconds. Values at or after contact are omitted.

    Returns:
        A pair containing intermediate ``(time, (x, y))`` points and the final
        ``(time_to_contact, focus_of_expansion)`` point.
    """
    if not np.isfinite(time_to_contact) or time_to_contact <= 0:
        raise ValueError("time_to_contact must be a finite positive number")
    camera_center = np.asarray(camera_center, dtype=float)
    focus_of_expansion = np.asarray(focus_of_expansion, dtype=float)
    if camera_center.shape != (2,) or focus_of_expansion.shape != (2,):
        raise ValueError("camera_center and focus_of_expansion must each contain two values")
    points = [
        (time, tuple(camera_center + (focus_of_expansion - camera_center) * (time / time_to_contact)))
        for time in ticks
        if 0 <= time < time_to_contact
    ]
    return points, (float(time_to_contact), tuple(focus_of_expansion))
