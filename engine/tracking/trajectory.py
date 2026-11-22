"""Short-horizon trajectory prediction with exponentially decaying acceleration.

This is a deterministic kinematic ODE model, not a neural network.
"""

import numpy as np
from scipy.integrate import solve_ivp


ACCEL_DECAY_TAU = 0.4

HORIZONS = [0.5, 1.0, 2.0, 3.0]
PREDICTION_STEPS = 200

MAX_PREDICTION_TIME = max(HORIZONS)


class Trajectory:
    def __init__(
        self, acceleration, runge_kutta_method: str = "RK45", max_time: float = None
    ):
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        if self.acceleration.shape != (2,) or not np.isfinite(self.acceleration).all():
            raise ValueError("acceleration must contain two finite values")
        self.runge_kutta_method = runge_kutta_method
        self.max_time = float(max_time) if max_time is not None else MAX_PREDICTION_TIME
        if not np.isfinite(self.max_time) or self.max_time <= 0:
            raise ValueError("max_time must be a finite positive number")

    def ode(self, t, state):
        _x, _y, velocity_x, velocity_y = state
        acceleration_x, acceleration_y = self.acceleration
        acceleration_decay = np.exp(-t / ACCEL_DECAY_TAU)
        return [
            velocity_x,
            velocity_y,
            acceleration_x * acceleration_decay,
            acceleration_y * acceleration_decay,
        ]

    def predict_trajectory(self, position, velocity):
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("position and velocity must each contain two values")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("position and velocity must contain only finite values")
        initial_state = [position[0], position[1], velocity[0], velocity[1]]

        times = np.linspace(0, self.max_time, PREDICTION_STEPS)

        solution = solve_ivp(
            self.ode,
            (0, self.max_time),
            initial_state,
            t_eval=times,
            method=self.runge_kutta_method,
        )

        if not solution.success:
            raise RuntimeError(f"ODE solve failed: {solution.message}")

        return (solution.t, solution.y[0], solution.y[1])

    def predicted_position(self, times, x, y, horizon):
        if horizon > times[-1] + 1e-9:
            raise ValueError(f"horizon {horizon}s longer than the ODE curve ({times[-1]}s)")
        px = np.interp(horizon, times, x)

        py = np.interp(horizon, times, y)

        return np.array([px, py])
