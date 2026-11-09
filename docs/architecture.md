# Architecture

Hawkeye-Zero processes a video one frame at a time through the `Tracker`
orchestrator in `engine/tracker.py`.

## Processing pipeline

1. OpenCV reads a frame and derives its timestamp from the frame index and
   video's frame rate.
2. The Ultralytics YOLO model selects the highest-confidence detection matching
   the configured object class.
3. The object center is added to the observation history. If camera
   compensation is enabled, the center is first transformed into stabilized
   coordinates.
4. After five observations are available, `StateEstimator` fits a quadratic
   curve to recent positions. It smooths velocity and acceleration estimates
   and limits acceleration magnitude.
5. `Trajectory` numerically integrates the estimated state with SciPy's
   `solve_ivp`, using exponentially decaying acceleration.
6. The visualizer annotates the frame. `PredictionLoss` later compares each
   prediction with an interpolated observation at its target time.
7. The tracker writes the annotated video and CSV logs for detections,
   predictions, comparisons, and aggregate errors.

## Camera compensation

When enabled, `CameraMotion` estimates frame-to-frame motion from optical-flow
features and maintains a frame-to-world transform. Object observations and
predictions are measured in that stabilized coordinate system; rendered
positions are transformed back into the current frame. Dark frames and
unreliable motion estimates do not update the transform. A detected shot change
resets recent observations and pending comparisons so predictions do not cross
shot boundaries.

## Important limitations

- The tracker keeps only the best matching detection per frame; it is not a
  multi-object identity tracker.
- Predictions assume smooth motion over a short horizon. Abrupt maneuvers,
  occlusion, detector errors, and shot changes can reduce accuracy.
- Error metrics are computed only when a later observation exists at the
  prediction's target time.
- Model weights and video assets are supplied by the user and are not bundled.
