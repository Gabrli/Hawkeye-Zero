# Algorithms and mathematical model

This document describes what the current source code computes. It intentionally distinguishes implemented behavior from possible future extensions.

## 1. Object detection

**Implementation:** `engine/camera/detector.py`

Ultralytics YOLO returns bounding boxes, class IDs, and confidence scores. The wrapper filters detections by the configured class and returns the one with the largest confidence score.

For a box with corners `(x1, y1)` and `(x2, y2)`, its center is:

\[
 x_c = \frac{x_1+x_2}{2}, \qquad y_c = \frac{y_1+y_2}{2}.
\]

The confidence score is the model's detection score; it should not automatically be interpreted as a calibrated probability. Selecting the best box independently in each frame is not multi-object tracking and does not guarantee that the same physical object is selected across frames.

## 2. Observation history

**Implementation:** `engine/tracking/history.py`

The history stores timestamps and 2D center positions. A short rolling window is used for state estimation, while a longer full history is retained for delayed evaluation.

`position_at(t)` uses linear interpolation between surrounding observations:

\[
 p(t) = (1-\lambda)p_i + \lambda p_{i+1}, \quad
 \lambda = \frac{t-t_i}{t_{i+1}-t_i}.
\]

It returns no position when `t` lies outside the observed time range. This is interpolation, not extrapolation.

## 3. Kinematic state estimation

**Implementation:** `engine/tracking/state.py`

The estimator fits a separate quadratic polynomial to x and y over the recent observations. Time is shifted so the latest observation is at zero:

\[
 \Delta t_i = t_i - t_{\mathrm{latest}}.
\]

For each coordinate:

\[
 q(\Delta t) = c_0 + c_1\Delta t + c_2\Delta t^2.
\]

At the latest observation, the estimates are:

\[
 position = c_0, \qquad velocity = c_1, \qquad acceleration = 2c_2.
\]

The coefficients are obtained with `numpy.polyfit`, which performs a least-squares polynomial fit. The acceleration vector is magnitude-clamped to `max_accel_magnitude`.

Velocity and acceleration are then smoothed independently using an exponential moving average:

\[
 s_t = \alpha x_t + (1-\alpha)s_{t-1}.
\]

The configured smoothing values are weights for the new measurement. Larger values respond faster; smaller values produce stronger smoothing.

## 4. Trajectory prediction

**Implementation:** `engine/tracking/trajectory.py`

The predictor starts from the latest estimated position and velocity. Its acceleration decreases exponentially:

\[
 a(t)=a_0 e^{-t/\tau_a},
\]

where `a0` is the estimated acceleration vector and `tau_a` is the fixed `ACCEL_DECAY_TAU` constant.

The integrated state is:

\[
 \frac{dx}{dt}=v_x,\quad \frac{dy}{dt}=v_y,
\]
\[
 \frac{dv_x}{dt}=a_{0x}e^{-t/\tau_a},\quad
 \frac{dv_y}{dt}=a_{0y}e^{-t/\tau_a}.
\]

SciPy's `solve_ivp` integrates this system using the configured method (`RK45` by default). The solution is sampled at a fixed set of output times. A requested horizon is read from that sampled curve using one-dimensional linear interpolation.

This is a conventional deterministic ODE model. There is no neural network, learned acceleration field, or online training in this implementation.

## 5. Prediction evaluation

**Implementation:** `engine/tracking/loss.py`, `engine/utils/calc_loss.py`

Each prediction stores its creation time, horizon, predicted position, and camera scale. When the video reaches the target time, the actual position is interpolated from observation history.

The Euclidean position error is:

\[
 e_i = \sqrt{(x_i^{pred}-x_i^{real})^2+(y_i^{pred}-y_i^{real})^2}.
\]

The aggregate metrics are:

\[
 MAE = \frac{1}{N}\sum_{i=1}^{N}|e_i|,
 \qquad
 RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}.
\]

The comparison is possible only if a later observation exists at the target timestamp. Missing observations are not counted as zero error.

## 6. Camera-motion compensation

**Implementation:** `engine/camera/camera_motion.py`

The camera-motion estimator:

1. Converts the frame to grayscale and downsamples it for processing.
2. Builds a mask that excludes borders, optional static HUD regions, and the most recent object bounding box.
3. Checks that enough image texture is available.
4. Selects corners with `cv2.goodFeaturesToTrack`.
5. Tracks corners forward and backward with pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
6. Rejects tracks whose forward-backward return error is too large.
7. Fits a partial affine transform with RANSAC (`cv2.estimateAffinePartial2D`).
8. Accepts the transform only if the inlier count and inlier ratio pass configured thresholds.

The accepted affine transform is converted back to full-resolution coordinates. The per-frame transform maps previous-frame points to current-frame points. The cumulative transform is updated by composing the inverse step so it maps current-frame coordinates into the stabilized coordinate system.

The scale estimate is derived from the 2x2 linear part `A`:

\[
 scale = \sqrt{|\det(A)|}.
\]

This is a 2D image scale estimate, not a metric distance or calibrated camera zoom measurement. A large mean difference between small frame thumbnails is used as a simple shot-change heuristic.

## 7. Focus of expansion and time-to-contact

**Implementation:** `engine/foe/estimator.py`

This component consumes reliable frame-to-frame transforms. For a step with linear part `A` and translation `b`, the FOE relation is:

\[
 (I-A)f=b.
\]

Rather than invert each individual matrix, the implementation combines recent steps into a least-squares normal equation:

\[
 \left(\sum_i B_i^TB_i\right)f = \sum_i B_i^Tb_i,
 \qquad B_i=I-A_i.
\]

Residuals are calculated for each step. A median-absolute-deviation (MAD) threshold removes inconsistent observations, and the least-squares estimate is repeated over a small fixed number of iterations. The resulting FOE and time-to-contact estimates are smoothed with EMA.

For an accepted expansion scale `m > 1`, the time-to-contact approximation is:

\[
 \tau = \frac{\Delta t}{1-1/m}.
\]

This approximation assumes that apparent expansion is dominated by forward motion toward a static scene. Rotation, lateral motion, independently moving objects, perspective variation, and inaccurate camera transforms can violate that assumption. The estimator is not a general-purpose landing detector.

## 8. Terrain segmentation

**Implementation:** `engine/camera/segmentation.py`

The segmenter uses the L channel of the Lab color space and several Gaussian-blur differences to build simple image features:

- A high-frequency response acts as a local texture / roughness proxy.
- A bright local-contrast response is used to propose blob-like candidates.
- A difference between two blur scales highlights darker local regions.

A threshold assigns initial labels. Bright candidates are cleaned with morphological opening, grouped with connected-component analysis, and filtered by area, aspect ratio, and fill ratio. Accepted components are expanded slightly with dilation and assigned the rock label. A median filter removes small isolated label regions.

These labels are heuristic image categories. They do not identify geological materials and are sensitive to lighting, shadows, camera resolution, and threshold settings.

## 9. Visualization and logging

**Implementation:** `engine/utils/visualizer.py`, `engine/utils/subtle.py`, `engine/utils/csv_logger.py`

The visualizer draws detections, observed trails, prediction curves, horizon markers, and a HUD. The subtle style uses a copied overlay and alpha blending to reduce visual prominence. CSV output is handled by a small logger that writes a header followed by rows.
