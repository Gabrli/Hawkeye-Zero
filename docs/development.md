# Development and testing

## Setup

Use Python 3.10 or newer. From the repository root:

```shell
python -m pip install -e ".[test]"
```

The package's runtime dependencies are declared in `pyproject.toml`. Tests use
pytest.

## Run tests

```shell
python -m pytest
```

`tests/unit/` covers the history, state estimation, trajectory, and prediction
loss components. `tests/e2e/` exercises the tracker from video capture through
CSV output using deterministic synthetic frames and a stub detector. The
end-to-end test does not download model weights or depend on external video
assets.

## Import and dependency checks

Use package imports such as `from engine import Tracker` or
`from engine.history import ObservationHistory`. Runtime third-party
dependencies are listed in `[project].dependencies`; test-only dependencies
belong in `[project.optional-dependencies].test`.

## Contributions

Keep changes focused, add or update tests for behavior changes, and update the
README or relevant documentation when user-facing behavior changes. Before
opening a contribution, run the full pytest suite and verify that the package
can be installed in editable mode.
