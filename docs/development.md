# Development and testing

## Environment

Use Python 3.10 or newer. Create a virtual environment and install the package with its test extra:

```bash
python -m pip install -e ".[test]"
```

The runtime dependencies are declared in `pyproject.toml`. The tests use pytest and deterministic synthetic data. The end-to-end test replaces the detector and video capture / writer with test doubles, so it does not download YOLO weights or require a real input video.

## Run tests

```bash
python -m pytest
```

Run a narrower test group while working on one component:

```bash
python -m pytest tests/unit/test_state.py
python -m pytest tests/e2e/
```

## Code organization

- Put camera and image-space sensing code in `engine/camera/`.
- Put observation history, state estimation, trajectory prediction, evaluation, and orchestration in `engine/tracking/`.
- Keep the descent FOE estimator separate in `engine/foe/`.
- Put reusable output and visualization helpers in `engine/utils/`.
- Prefer descriptive local names such as `previous_points`, `current_points`, `forward_backward_error`, `world_from_frame`, and `time_to_contact` over short names whose meaning depends on a comment.

## Testing expectations

When changing an algorithm:

1. Add a small deterministic unit test for the mathematical behavior.
2. Add an edge-case test for invalid shapes, times, or horizons where applicable.
3. Update the relevant algorithm or API documentation.
4. Run the full test suite.

Do not label a model as neural, learned, metric, or physically calibrated unless the implementation actually provides those properties.

## Packaging

The package is discovered from `engine*` by setuptools. Keep runtime dependencies in `[project].dependencies` and test-only dependencies in `[project.optional-dependencies].test`. Before publishing, build and install a wheel in a clean environment and verify the documented CLI and Python imports.
