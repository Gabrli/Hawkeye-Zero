# API reference

The public convenience imports are available from `engine`. More specific imports should use the package paths below.

## `Tracker`

**Import:** `from engine import Tracker`

`Tracker(video_path, ...)` opens the input video and prepares the detector, history, estimator, prediction evaluator, visualizer, and CSV writers.

Important arguments:

| Argument | Meaning |
| --- | --- |
| `video_path` | Input video path. |
| `output_video` | Annotated video output path. |
| `output_detections` | Detection CSV path. |
| `output_predictions` | Prediction CSV path. |
| `output_comparison` | Delayed prediction comparison CSV path. |
| `output_metrics` | Aggregate metrics CSV path. |
| `model_path` | Ultralytics model weights path. |
| `object_class` | Class ID to retain. |
| `confidence` | Minimum YOLO confidence threshold. |
| `horizons` | Future prediction times in seconds. |
| `camera_compensation` | Enable background-motion compensation. |
| `output_camera` | Camera transform CSV path. |
| `camera` | Optional preconfigured `CameraMotion` instance. |
| `prediction_color` | Prediction overlay color in OpenCV BGR order. |
| `hud_alpha` | Darkness strength for the HUD background. |
| `compact_hud` | Use a compact HUD layout. |
| `imgsz` | Optional YOLO inference image size. |
| `prediction_style` | `"bold"` or `"subtle"` overlay style. |

Methods:

- `process_frame(frame, current_time)`: process and annotate one BGR frame in place.
- `run(display=True)`: process the full video. When `display` is true, show a preview window; press `Esc` to stop.

## `Detection` and `Yolo`

**Import:** `from engine.camera.detector import Detection, Yolo`

`Detection` contains `cx`, `cy`, `x1`, `y1`, `x2`, `y2`, and `confidence`.

`Yolo(model_path="yolo11n.pt", object_class=0, imgsz=None)` loads an Ultralytics model. `detect(frame, conf, verbose=False)` returns the highest-confidence matching `Detection`, or `None` if no matching detection exists. The `ultralytics` dependency is imported when a `Yolo` instance is created.

## `CameraMotion`

**Import:** `from engine.camera.camera_motion import CameraMotion, CameraState`

`CameraMotion.update(frame, exclude_bbox=None)` processes a BGR frame and returns a `CameraState`:

- `W`: cumulative current-frame-to-stabilized-coordinate transform (3x3 homogeneous matrix).
- `reliable`: whether a valid motion estimate was accepted for this update.
- `cut`: whether a shot change was detected.
- `inliers`: number of RANSAC inliers in the accepted / attempted estimate.
- `scale`: scale of the cumulative transform.
- `step`: previous-frame-to-current-frame transform for this update.

`reset()` clears the transform and previous-frame state. `build_static_hud_mask(video_path, ...)` creates a mask of persistent high-frequency screen overlays from sampled frames.

## `TerrainSegmenter`

**Import:** `from engine.camera.segmentation import TerrainSegmenter`

`features(bgr_frame)` returns three feature maps: texture response, bright-blob response, and dark-region response.

`segment(bgr_frame)` returns `(labels, rock_boxes, class_fractions)`. Labels are integer class IDs in the range 0–3. Each rock box is `(x1, y1, x2, y2, area)` in the input frame's approximate pixel coordinates. Class fractions are the proportion of pixels assigned to each class.

## `DescentEstimator`

**Import:** `from engine.foe.estimator import DescentEstimator, trajectory_points`

`DescentEstimator(fps=30.0, window_s=1.5, smooth=0.15, min_steps=10)` accumulates reliable expansion transforms. `update(step, reliable)` accepts a 3x3 previous-frame-to-current-frame transform and a reliability flag. It returns `(focus_of_expansion, time_to_contact_seconds)` when enough consistent data is available, otherwise `None` until an estimate has been established.

`reset()` clears accumulated transforms and smoothed estimates.

`trajectory_points(center, foe, tau, ticks=...)` returns intermediate image-space points along the line from the camera center to the FOE and the final FOE point at the estimated time-to-contact.

## `ObservationHistory`

**Import:** `from engine.tracking.history import ObservationHistory`

- `add(time, x, y)`: append one observation.
- `recent_points()` / `recent_times()`: return the rolling window used by the state estimator.
- `position_at(target_time)`: linearly interpolate a position inside the observed time range, or return `None` outside it.
- `reset()`: clear recent and full history.

Observation timestamps should be strictly increasing.

## `StateEstimator`

**Import:** `from engine.tracking.state import StateEstimator`

`estimate(points, times)` returns `(position, velocity, acceleration)` as NumPy arrays, or `None` when fewer than five points are available. The estimator fits independent quadratic polynomials for x and y, smooths velocity / acceleration, and clamps acceleration magnitude. `reset()` clears the smoothing state.

## `Trajectory`

**Import:** `from engine.tracking.trajectory import Trajectory`

`Trajectory(acceleration, runge_kutta_method="RK45", max_time=None)` creates the decaying-acceleration ODE model.

- `ode(time, state)`: derivative for `[x, y, vx, vy]`.
- `predict_trajectory(position, velocity)`: integrate the ODE and return `(times, x_positions, y_positions)`.
- `predicted_position(times, x_positions, y_positions, horizon)`: interpolate the predicted position at a requested horizon. Raises `ValueError` if the horizon exceeds the computed curve.

## `PredictionLoss`

**Import:** `from engine.tracking.loss import PredictionLoss`

`PredictionLoss(horizons, history)` stores pending predictions and completed errors. `register(...)` queues a prediction, `resolve(current_time)` evaluates predictions whose target times have arrived, `mae(horizon)` and `rmse(horizon)` return aggregate errors, and `reset()` discards pending predictions while preserving completed error samples.
