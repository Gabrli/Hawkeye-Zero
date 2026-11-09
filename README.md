# Hawkeye-Zero

Hawkeye-Zero detects a selected object in a video, estimates its motion, and
draws a predicted trajectory over the footage. It uses Ultralytics YOLO for
detection, a smoothed kinematic estimate for velocity and acceleration, and
SciPy's ODE solver to predict future positions. Optional camera-motion
compensation expresses observations in a stabilized coordinate system.

## Features

- Select the object class and confidence threshold.
- Predict positions at configurable time horizons.
- Export detections, predictions, prediction comparisons, and error metrics as
  CSV files.
- Optionally compensate for camera movement and detect shot changes.
- Render the result with a configurable prediction color and HUD.

## Requirements

- Python 3.10 or newer.
- A video file readable by OpenCV.
- A YOLO model compatible with Ultralytics. The default model is
  `yolo11n.pt`; Ultralytics downloads it when it is not already available.

## Installation

From the repository root, create and activate a virtual environment, then
install the project:

```shell
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```shell
source .venv/bin/activate
```

Install the package and its test dependencies:

```shell
python -m pip install -e ".[test]"
```

## Run

Pass a video file and, optionally, a model weights file:

```shell
python run_tracker.py path/to/video.mp4 --weights path/to/model.pt
```

Use the built-in default model, show the preview window, choose a prediction
color, or change the output directory:

```shell
python run_tracker.py path/to/video.mp4 --show --color yellow --output-dir outputs
```

Run `python run_tracker.py --help` for all available options. Output files are
created in the selected output directory:

| File | Contents |
| --- | --- |
| `output.mp4` | Annotated video |
| `detections.csv` | Per-frame detections and bounding boxes |
| `predictions.csv` | Predicted positions and estimated motion state |
| `trajectory_comparison.csv` | Predictions compared with later observations |
| `metrics.csv` | Sample count, MAE, and RMSE per horizon |
| `camera.csv` | Camera transform data; only when camera compensation is enabled |

The Python API also supports camera compensation:

```python
from engine import Tracker

tracker = Tracker(
    "path/to/video.mp4",
    model_path="path/to/model.pt",
    camera_compensation=True,
)
tracker.run(display=False)
```

The input video and model weights are not included. Check the terms for your
video data and model before using or redistributing them.

## Development

Run the unit and end-to-end tests:

```shell
python -m pytest
```

See [development and testing](docs/development.md) for test scope and
[architecture](docs/architecture.md) for the processing pipeline and coordinate
systems. Contributions should follow the [Code of Conduct](CODE_OF_CONDUCT.MD).

## License

This project is licensed under the GNU General Public License v3.0 only. See
[LICENSE](LICENSE).
