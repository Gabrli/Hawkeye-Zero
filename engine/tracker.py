import cv2
import numpy as np

from engine.camera_motion import CameraMotion
from engine.detector import Yolo
from engine.history import ObservationHistory
from engine.state import StateEstimator
from engine.trajectory import Trajectory
from engine.constants import HORIZONS, OBJECT_CLASS, MODEL_PATH
from engine.loss import PredictionLoss
from engine.visualizer import Visualizer
from engine.csv_logger import CsvLogger


class Tracker:
    """Run detection, motion estimation, trajectory prediction, and video export."""

    def __init__(
        self,
        video_path: str,
        output_video: str = "output.mp4",
        output_detections: str = "detections.csv",
        output_predictions: str = "predictions.csv",
        output_comparison: str = "trajectory_comparison.csv",
        output_metrics: str = "metrics.csv",
        model_path: str = MODEL_PATH,
        object_class: int = OBJECT_CLASS,
        confidence: float = 0.35,
        horizons=None,
        camera_compensation: bool = False,
        output_camera: str = "camera.csv",
        camera: CameraMotion = None,
        prediction_color=(0, 255, 255),
        hud_alpha: float = 1.0,
        compact_hud: bool = False,
    ):
        """Initialize video processing, model inference, and output logs.

        Args:
            video_path: Input video file path.
            output_video: Destination for the annotated video.
            output_detections: CSV destination for detected boxes and centers.
            output_predictions: CSV destination for predicted positions and states.
            output_comparison: CSV destination for resolved prediction errors.
            output_metrics: CSV destination for aggregate error metrics.
            model_path: Path or Ultralytics identifier for model weights.
            object_class: Model class index to track.
            confidence: Minimum detection confidence.
            horizons: Future prediction horizons in seconds.
            camera_compensation: Whether to estimate and compensate for camera motion.
            output_camera: CSV destination for camera transforms when enabled.
            camera: Optional preconfigured camera-motion estimator.
            prediction_color: BGR color used for trajectory predictions.
            hud_alpha: Opacity of the HUD background, from transparent to opaque.
            compact_hud: Whether to draw the compact HUD layout.

        Raises:
            RuntimeError: If OpenCV cannot open ``video_path``.
        """
        self.confidence = confidence
        self.horizons = horizons or HORIZONS
        self.output_metrics = output_metrics

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_video, fourcc, self.fps, (width, height))

        self.yolo = Yolo(model_path=model_path, object_class=object_class)
    
        self.history = ObservationHistory()
        self.estimator = StateEstimator()
        self.verifier = PredictionLoss(self.horizons, self.history)
        self.visualizer = Visualizer(prediction_color, hud_alpha, compact_hud)

        self.detections_log = CsvLogger(
            output_detections,
            ["frame", "time", "x", "y", "x1", "y1", "x2", "y2", "confidence"],
        )
        self.predictions_log = CsvLogger(
            output_predictions,
            ["frame", "time", "horizon", "x_pred", "y_pred", "vx", "vy", "ax", "ay"],
        )
        self.comparison_log = CsvLogger(
            output_comparison,
            ["frame", "prediction_time", "horizon", "target_time",
             "x_pred", "y_pred", "x_real", "y_real", "error_x", "error_y", "error"],
        )

    
        self.camera = camera if camera is not None else (CameraMotion() if camera_compensation else None)
        self._last_bbox = None
        if self.camera is not None:
            self.camera_log = CsvLogger(
                output_camera,
                ["frame", "time", "reliable", "cut", "inliers", "scale",
                 "w00", "w01", "w02", "w10", "w11", "w12"],
            )

        self.frame_id = 0

    def _reset_segment(self):
        """Clear motion and pending-prediction state at a shot boundary."""
        # new shot
        self.history.reset()
        self.estimator.reset()
        self.verifier.reset()

    def process_frame(self, frame, current_time: float):
        """Process a frame in place and write any resulting CSV records.

        Args:
            frame: BGR video frame to annotate.
            current_time: Timestamp of the frame in seconds.
        """
        W, Winv, scale, cam = None, None, 1.0, None
        if self.camera is not None:
            cam = self.camera.update(frame, self._last_bbox)
            if cam.cut:
                self._reset_segment()
            W, scale = cam.W, cam.scale
            Winv = np.linalg.inv(W)
            self.camera_log.write([
                self.frame_id, current_time, int(cam.reliable), int(cam.cut), cam.inliers, cam.scale,
                W[0, 0], W[0, 1], W[0, 2], W[1, 0], W[1, 1], W[1, 2],
            ])

        def to_frame(xs, ys):
            """Map stable-world coordinates back into the current frame."""
            if Winv is None:
                return xs, ys
            xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            return Winv[0, 0] * xs + Winv[0, 1] * ys + Winv[0, 2], Winv[1, 0] * xs + Winv[1, 1] * ys + Winv[1, 2]

        detection = self.yolo.detect(frame, conf=self.confidence)
        self._last_bbox = None if detection is None else (detection.x1, detection.y1, detection.x2, detection.y2)

        if detection is not None:
            hx, hy = detection.cx, detection.cy
            if W is not None:
                hx, hy = (W @ np.array([hx, hy, 1.0]))[:2]
            self.history.add(current_time, hx, hy)

            self.detections_log.write([
                self.frame_id, current_time, detection.cx, detection.cy,
                detection.x1, detection.y1, detection.x2, detection.y2,
                detection.confidence,
            ])

            self.visualizer.draw_detection(
                frame, detection.x1, detection.y1, detection.x2, detection.y2,
                detection.cx, detection.cy,
            )

        for resolved in self.verifier.resolve(current_time):
            self.comparison_log.write(list(resolved))

        if len(self.history) >= 5:
            state = self.estimator.estimate(self.history.recent_points(), self.history.recent_times())

            if state is not None:
                position, velocity, acceleration = state

                solver = Trajectory(acceleration, max_time=max(self.horizons))
                times, x_pred, y_pred = solver.predict_trajectory(position, velocity)

                cx_pred, cy_pred = to_frame(x_pred, y_pred)
                self.visualizer.draw_prediction_curve(frame, cx_pred, cy_pred)

                prev_draw, slot = None, 0
                for horizon in self.horizons:
                    predicted = solver.predicted_position(times, x_pred, y_pred, horizon)

                    px_draw, py_draw = to_frame(predicted[0], predicted[1])
                    # etykiety przesuwamy w górę tylko dla punktów blisko poprzedniego (unikamy nachodzenia)
                    if prev_draw is not None and np.hypot(px_draw - prev_draw[0], py_draw - prev_draw[1]) < 32:
                        slot += 1
                    else:
                        slot = 0
                    prev_draw = (float(px_draw), float(py_draw))
                    self.visualizer.draw_prediction_point(frame, (px_draw, py_draw), f"+{horizon:.1f}s", slot)

                    self.predictions_log.write([
                        self.frame_id, current_time, horizon,
                        predicted[0], predicted[1],
                        velocity[0], velocity[1],
                        acceleration[0], acceleration[1],
                    ])

                    self.verifier.register(self.frame_id, current_time, horizon, predicted)

        trail = self.history.recent_points()
        if Winv is not None and len(trail) > 0:
            tx, ty = to_frame([p[0] for p in trail], [p[1] for p in trail])
            trail = list(zip(tx, ty))
        self.visualizer.draw_history(frame, trail)
        self.visualizer.draw_hud(frame, current_time, self.verifier)
        if cam is not None:
            self.visualizer.draw_status(
                frame,
                "KAMERA: tlo zmierzone (ruch wzgl. tla)" if cam.reliable else "KAMERA: brak tla - zamrozona",
                ok=cam.reliable,
            )

    def run(self, display: bool = True):
        """Process the input video until it ends or the preview is closed.

        Args:
            display: Whether to show an interactive preview window.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time = self.frame_id / self.fps

            self.process_frame(frame, current_time)

            self.writer.write(frame)

            if display:
                cv2.imshow("YOLO + ODE Prediction", frame)
                if cv2.waitKey(1) == 27:
                    break

            self.frame_id += 1

        self._shutdown()

    def _shutdown(self):
        """Release video and CSV resources, then write final metrics."""
        self.cap.release()
        self.writer.release()
        self.detections_log.close()
        self.predictions_log.close()
        self.comparison_log.close()
        if self.camera is not None:
            self.camera_log.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # OpenCV bez GUI (headless)
        self._write_metrics()

    def _write_metrics(self):
        """Write per-horizon sample counts, MAE, and RMSE to the metrics CSV."""
        metrics_log = CsvLogger(
            self.output_metrics,
            ["horizon_seconds", "samples", "MAE_pixels", "RMSE_pixels"],
        )

        for horizon in self.horizons:
            n = len(self.verifier.errors[horizon])
            mae = self.verifier.mae(horizon)
            rmse = self.verifier.rmse(horizon)

            mae_out = mae if mae is not None else float("nan")
            rmse_out = rmse if rmse is not None else float("nan")

            metrics_log.write([horizon, n, mae_out, rmse_out])

            print(f"{horizon:.1f}s | N={n:4d} | MAE={mae_out:.2f}px | RMSE={rmse_out:.2f}px")

        metrics_log.close()