import numpy as np
from scipy.integrate import solve_ivp
from engine.constants import HORIZONS

ACCEL_DECAY_TAU = 0.4

PREDICTION_STEPS = 200

MAX_PREDICTION_TIME = max(HORIZONS)


class Trajectory:
    """Integrate a motion state forward with exponentially decaying acceleration."""

    def __init__(self, acceleration, runge_kutta_method: str = "RK45", max_time: float = None):
        """Configure the acceleration vector and integration interval.

        Args:
            acceleration: Two-element acceleration vector at time zero.
            runge_kutta_method: SciPy ``solve_ivp`` integration method.
            max_time: Maximum prediction time in seconds. Defaults to the
                largest configured horizon.
        """
        self.acceleration = np.asarray(acceleration, dtype=np.float64)
        self.runge_kutta_method = runge_kutta_method
        # max horizon
        self.max_time = float(max_time) if max_time is not None else MAX_PREDICTION_TIME

    def ode(self, t, state):
        """Evaluate the motion differential equation at a time and state.

        Args:
            t: Elapsed time in seconds.
            state: ``(x, y, vx, vy)`` position and velocity state.

        Returns:
            Derivatives ``(vx, vy, ax, ay)`` at time ``t``.
        """
        _, _, vx, vy = state

        ax, ay = self.acceleration

        decay = np.exp(-t / ACCEL_DECAY_TAU)

        return [vx, vy, ax * decay, ay * decay]

    def predict_trajectory(self, position, velocity):
        """Numerically integrate a trajectory over the configured interval.

        Args:
            position: Two-element initial ``(x, y)`` position.
            velocity: Two-element initial ``(vx, vy)`` velocity.

        Returns:
            Three arrays containing sample times, x positions, and y positions.

        Raises:
            RuntimeError: If SciPy's ODE solver does not complete successfully.
        """
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
        """Interpolate the predicted position at a requested future horizon.

        Args:
            times: Sample times returned by :meth:`predict_trajectory`.
            x: Predicted x coordinates at ``times``.
            y: Predicted y coordinates at ``times``.
            horizon: Requested elapsed time in seconds.

        Returns:
            A two-element NumPy array containing the interpolated position.

        Raises:
            ValueError: If ``horizon`` exceeds the computed trajectory.
        """
        if horizon > times[-1] + 1e-9:
            raise ValueError(f"horizon {horizon}s is longer than the ODE curve ({times[-1]}s)")
        px = np.interp(horizon, times, x)

        py = np.interp(horizon, times, y)

        return np.array([px, py])
