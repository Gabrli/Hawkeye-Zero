# Hawkeye-Zero

Hawkeye-Zero is a Python video-analysis project that detects one selected object, estimates its recent 2D motion, predicts short-term future positions, and writes an annotated video with CSV reports.

The project also contains reusable components for camera-motion compensation, focus-of-expansion (FOE) estimation, and heuristic terrain segmentation. These components are available as separate modules; FOE estimation and terrain segmentation are not currently part of the default `Tracker` processing loop.

> **Model note:** The trajectory predictor is a deterministic kinematic ODE with exponentially decaying acceleration. It is not a Neural ODE and does not train a neural network.

## Features

- Detect the highest-confidence object of a selected YOLO class in each frame.
- Keep a time-stamped history of detected object centers.
- Estimate position, velocity, and acceleration from a quadratic least-squares fit.
- Smooth velocity and acceleration estimates with exponential moving averages (EMA).
- Predict future positions by integrating a small ODE with SciPy's `solve_ivp` (RK45 by default).
- Optionally compensate for camera movement using background optical flow and a robust affine transform.
- Estimate a focus of expansion and time-to-contact with the standalone `DescentEstimator`.
- Segment terrain using image-processing heuristics (texture, bright blobs, dark regions, and connected components).
- Save detections, predictions, prediction errors, metrics, and optional camera transforms as CSV files.

## Requirements

- Python 3.10 or newer.
- A video file supported by OpenCV.
- A compatible Ultralytics YOLO model. The default weights path is `yolo11n.pt`.
- Runtime libraries listed in `pyproject.toml` (`numpy`, `opencv-python`, `scipy`, `torch`, and `ultralytics`).

The model weights and input videos are not included. Ultralytics may download the default weights on first use. Check the terms for your video data and model weights before using or redistributing them.

## Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the package and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Quick start

Process a video with the default model and default options:

```bash
python run_tracker.py path/to/video.mp4
```

Show a preview window and choose an output directory:

```bash
python run_tracker.py path/to/video.mp4 --show --output-dir outputs
```

Use custom weights, class, confidence threshold, prediction horizons, and camera compensation:

```bash
python run_tracker.py path/to/video.mp4 \
  --weights path/to/model.pt \
  --class-id 0 \
  --confidence 0.35 \
  --horizons 0.5 1 2 3 \
  --camera-compensation \
  --output-dir outputs
```

Run `python run_tracker.py --help` to see all command-line options. Press `Esc` to stop preview mode early.

## Python API

```python
from engine import Tracker

tracker = Tracker(
    video_path="path/to/video.mp4",
    model_path="yolo11n.pt",
    object_class=0,
    confidence=0.35,
    horizons=[0.5, 1.0, 2.0],
    camera_compensation=True,
    output_video="outputs/annotated.mp4",
    output_detections="outputs/detections.csv",
    output_predictions="outputs/predictions.csv",
    output_comparison="outputs/comparisons.csv",
    output_metrics="outputs/metrics.csv",
    output_camera="outputs/camera.csv",
)
tracker.run(display=False)
```

The reusable components can also be imported separately:

```python
from engine.camera.camera_motion import CameraMotion
from engine.camera.segmentation import TerrainSegmenter
from engine.foe.estimator import DescentEstimator, trajectory_points
from engine.tracking.history import ObservationHistory
from engine.tracking.state import StateEstimator
from engine.tracking.trajectory import Trajectory
```

See [API reference](docs/api.md) for constructor arguments, return values, and coordinate conventions.

## Output files

The output directory is created by the command-line entry point. The tracker writes:

| File | Description |
| --- | --- |
| `output.mp4` | Annotated video. |
| `detections.csv` | One selected detection per frame, including center, bounding box, and confidence. |
| `predictions.csv` | Predicted coordinates and the estimated velocity / acceleration for each horizon. |
| `trajectory_comparison.csv` | Predictions compared with an interpolated later observation. |
| `metrics.csv` | Number of evaluated predictions, MAE, and RMSE for each horizon. |
| `camera.csv` | Camera reliability, inlier count, scale, and cumulative transform; only written when camera compensation is enabled. |

Coordinates and error metrics are in pixels or pixel-equivalent stabilized coordinates. They are not metric-world measurements.

## Documentation

- [Architecture and data flow](docs/architecture.md)
- [Algorithms and mathematical model](docs/algorithms.md)
- [Python API reference](docs/api.md)
- [Development and testing](docs/development.md)

## Testing

Run the complete test suite:

```bash
python -m pytest
```

The end-to-end test uses synthetic frames and a stub detector. It does not require model weights or a real video file.

## Known limitations

- Only the highest-confidence detection for the configured class is selected per frame. There is no multi-object identity association or persistent object ID.
- The predictor assumes smooth short-term motion. It cannot anticipate abrupt changes that are not present in the recent observations.
- The acceleration-decay time constant is a fixed heuristic, not a learned physical parameter.
- Camera compensation is based on a 2D similarity-like affine estimate of visible background motion. Parallax, rolling shutter, moving backgrounds, low texture, and large scene changes can make it unreliable.
- The FOE / time-to-contact estimator assumes expansion caused by forward motion and is not a general 3D landing estimator.
- Terrain classes are heuristic image labels, not geological or semantic classifications.
- MAE and RMSE are calculated only when a later observation is available at the requested prediction time.
- The default tracker pipeline does not currently call the terrain segmenter or the FOE descent estimator.

## License

This project is licensed under the GNU General Public License v3.0 only. See [LICENSE](LICENSE).
