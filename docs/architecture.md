# Architecture

Hawkeye-Zero is organized around small processing components and a `Tracker` orchestrator. The main implementation is under `engine/`; `run_tracker.py` is the command-line entry point.

## Package layout

```text
engine/
├── camera/
│   ├── camera_motion.py   # background optical flow and camera transforms
│   ├── detector.py        # YOLO wrapper and Detection value object
│   └── segmentation.py    # heuristic terrain segmentation
├── constants/             # default class and prediction settings
├── foe/
│   └── estimator.py       # focus of expansion and time-to-contact
├── tracking/
│   ├── history.py         # timestamped object observations
│   ├── loss.py            # delayed prediction evaluation
│   ├── state.py           # position / velocity / acceleration estimate
│   ├── trajectory.py      # ODE integration and future positions
│   └── tracker.py         # main video processing pipeline
└── utils/
    ├── calc_loss.py       # MAE and RMSE
    ├── csv_logger.py      # CSV output helper
    ├── subtle.py          # low-contrast drawing primitives
    └── visualizer.py      # video overlays and HUD
```

A few top-level modules (for example `engine.history` and `engine.trajectory`) remain as compatibility imports. New code should prefer the package paths shown above.

## Main tracking pipeline

For each input frame, `engine.tracking.tracker.Tracker` performs the following steps:

1. Read the frame and derive its timestamp as `frame_index / fps`.
2. If camera compensation is enabled, estimate background motion and update the transform between current-frame coordinates and stabilized coordinates.
3. Run YOLO and select the highest-confidence detection matching the configured class.
4. Transform the detected center into stabilized coordinates when camera compensation is active, then append it to `ObservationHistory`.
5. Resolve predictions created earlier if their target timestamps have now been reached. The ground-truth position is linearly interpolated from the full observation history.
6. Once at least five recent observations exist, fit a quadratic curve to the recent x and y coordinates. Smooth the derived velocity and acceleration estimates.
7. Integrate the trajectory ODE from the latest estimated position and velocity. Draw the curve and requested horizon markers.
8. Register each prediction for later evaluation and write detections, predictions, and comparisons to CSV.
9. Write the annotated frame to the output video.

## Coordinate systems

Without camera compensation, all positions are expressed in the current image's pixel coordinates.

With camera compensation enabled:

- `CameraState.step` maps points from the previous frame to the current frame.
- `CameraState.W` maps points from the current frame to a stabilized coordinate system.
- Detected centers are transformed by `W` before entering the observation history.
- Predicted stabilized points and history trails are transformed back into the current frame for drawing.
- Prediction errors are divided by the cumulative scale so reported errors remain approximately comparable to current-frame pixel distances.

A shot change resets the camera transform and the tracker's recent observation / pending-prediction state. Completed error samples are retained.

## Independent components

The following components are implemented but are not called by the default `Tracker` loop:

- `engine.foe.estimator.DescentEstimator`: estimates a focus of expansion and a time-to-contact from reliable frame-to-frame transforms.
- `engine.camera.segmentation.TerrainSegmenter`: creates heuristic terrain labels and candidate rock regions.

They can be used independently or integrated into a future application layer.

## Design boundaries

- `camera/` contains image-space sensing and image processing.
- `tracking/` contains object observation history, kinematic estimation, prediction, evaluation, and orchestration.
- `foe/` contains the separate descent-geometry estimator.
- `utils/` contains output and visualization helpers.

The detector is currently a thin wrapper around Ultralytics YOLO. The tracking pipeline does not depend on YOLO-specific result objects after conversion to the small `Detection` value object.

## Failure handling

- If a frame contains no matching detection, no observation is added for that frame.
- The state estimator returns no state until enough observations are available or if its fit is invalid.
- A failed ODE integration raises a `RuntimeError` rather than returning a partial curve as if it were valid.
- An unreliable camera estimate leaves the cumulative transform unchanged.
- Predictions are evaluated only when a real observation can be interpolated at the prediction's target time.

## Limitations

This is a 2D image-space predictor, not a calibrated 3D motion-reconstruction system. It has no object identity tracker, camera calibration, depth model, terrain-aware trajectory planner, or uncertainty model. The FOE estimator uses assumptions about forward expansion and should not be interpreted as a general landing solution.
