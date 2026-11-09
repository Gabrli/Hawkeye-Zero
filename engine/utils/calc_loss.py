import numpy as np


EQUATIONS = {
    "mae": lambda err: np.mean(np.abs(err)),
    "rmse": lambda err: np.sqrt(np.mean(np.square(err))),
}


def calc_error(errors, metric_type):
    """Calculate a supported aggregate error metric.

    Args:
        errors: Numeric error values.
        metric_type: Metric name, either ``"mae"`` or ``"rmse"``.

    Returns:
        The requested metric as a NumPy scalar, or NaN when ``errors`` is empty.

    Raises:
        ValueError: If ``metric_type`` is not supported.
    """
    errors = np.asarray(errors, dtype=np.float64)

    if len(errors) == 0:
        return np.nan

    if metric_type not in EQUATIONS:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    return EQUATIONS[metric_type](errors)
 