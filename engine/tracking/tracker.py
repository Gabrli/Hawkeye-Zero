import cv2
import numpy as np

from engine.camera.camera_motion import CameraMotion
from engine.camera.detector import Yolo
from engine.constants import HORIZONS
from engine.tracking.history import ObservationHistory
from engine.tracking.state import StateEstimator
from engine.tracking.trajectory import Trajectory
from engine.tracking.loss import PredictionLoss
from engine.utils.visualizer import Visualizer
from engine.utils.csv_logger import CsvLogger


class Tracker:
    """Run detection, state estimation, trajectory prediction, and video output."""

    def __init__(
        self,
        video_path: str,
        output_video: str = "output.mp4",
        output_detections: str = "detections.csv",
        output_predictions: str = "predictions.csv",
        output_comparison: str = "trajectory_comparison.csv",
        output_metrics: str = "metrics.csv",
        model_path: str = "yolo11n.pt",
        object_class: int = 0,
        confidence: float = 0.35,
        horizons=None,
        camera_compensation: bool = False,
        output_camera: str = "camera.csv",
        camera: CameraMotion = None,
        prediction_color=(0, 255, 255),
        hud_alpha: float = 1.0,
        compact_hud: bool = False,
        imgsz: int = None,
        prediction_style: str = "bold",
    ):
        self.confidence = confidence
        self.horizons = horizons or HORIZONS
        self.output_metrics = output_metrics

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can not open {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_video, fourcc, self.fps, (width, height))

        self.yolo = Yolo(model_path=model_path, object_class=object_class, imgsz=imgsz)

        self.history = ObservationHistory()
        self.estimator = StateEstimator()
        self.verifier = PredictionLoss(self.horizons, self.history)
        self.visualizer = Visualizer(
            prediction_color, hud_alpha, compact_hud, style=prediction_style
        )

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
            [
                "frame",
                "prediction_time",
                "horizon",
                "target_time",
                "x_pred",
                "y_pred",
                "x_real",
                "y_real",
                "error_x",
                "error_y",
                "error",
            ],
        )

        self.camera = (
            camera
            if camera is not None
            else (CameraMotion() if camera_compensation else None)
        )
        self._last_bbox = None
        if self.camera is not None:
            self.camera_log = CsvLogger(
                output_camera,
                [
                    "frame",
                    "time",
                    "reliable",
                    "cut",
                    "inliers",
                    "scale",
                    "w00",
                    "w01",
                    "w02",
                    "w10",
                    "w11",
                    "w12",
                ],
            )

        self.frame_id = 0

    def _reset_segment(self):
        self.history.reset()
        self.estimator.reset()
        self.verifier.reset()

    def process_frame(self, frame, current_time: float):
        world_from_frame = None
        frame_from_world = None
        camera_scale = 1.0
        camera_state = None
        if self.camera is not None:
            camera_state = self.camera.update(frame, self._last_bbox)
            if camera_state.cut:
                self._reset_segment()
            world_from_frame, camera_scale = camera_state.world_from_frame, camera_state.scale
            frame_from_world = np.linalg.inv(world_from_frame)
            self.camera_log.write(
                [
                    self.frame_id,
                    current_time,
                    int(camera_state.reliable),
                    int(camera_state.cut),
                    camera_state.inliers,
                    camera_state.scale,
                    world_from_frame[0, 0],
                    world_from_frame[0, 1],
                    world_from_frame[0, 2],
                    world_from_frame[1, 0],
                    world_from_frame[1, 1],
                    world_from_frame[1, 2],
                ]
            )

        def to_frame(xs, ys):
            if frame_from_world is None:
                return xs, ys
            xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            return (
                frame_from_world[0, 0] * xs + frame_from_world[0, 1] * ys + frame_from_world[0, 2],
                frame_from_world[1, 0] * xs + frame_from_world[1, 1] * ys + frame_from_world[1, 2],
            )

        detection = self.yolo.detect(frame, conf=self.confidence)
        self._last_bbox = (
            None
            if detection is None
            else (detection.x1, detection.y1, detection.x2, detection.y2)
        )

        if detection is not None:
            history_x, history_y = detection.cx, detection.cy
            if world_from_frame is not None:
                history_x, history_y = (world_from_frame @ np.array([history_x, history_y, 1.0]))[:2]
            self.history.add(current_time, history_x, history_y)

            self.detections_log.write(
                [
                    self.frame_id,
                    current_time,
                    detection.cx,
                    detection.cy,
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    detection.confidence,
                ]
            )

            self.visualizer.draw_detection(
                frame,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                detection.cx,
                detection.cy,
            )

        for resolved in self.verifier.resolve(current_time):
            self.comparison_log.write(list(resolved))

        if len(self.history) >= 5:
            estimated_state = self.estimator.estimate(
                self.history.recent_points(), self.history.recent_times()
            )

            if estimated_state is not None:
                position, velocity, acceleration = estimated_state

                solver = Trajectory(acceleration, max_time=max(self.horizons))
                prediction_times, predicted_x_positions, predicted_y_positions = (
                    solver.predict_trajectory(position, velocity)
                )

                predicted_curve_x, predicted_curve_y = to_frame(predicted_x_positions, predicted_y_positions)
                self.visualizer.draw_prediction_curve(frame, predicted_curve_x, predicted_curve_y)

                previous_marker_position, label_slot = None, 0
                for horizon in self.horizons:
                    predicted = solver.predicted_position(
                        prediction_times, predicted_x_positions, predicted_y_positions, horizon
                    )

                    predicted_frame_x, predicted_frame_y = to_frame(predicted[0], predicted[1])

                    if previous_marker_position is not None and np.hypot(
                        predicted_frame_x - previous_marker_position[0], predicted_frame_y - previous_marker_position[1]
                    ) < 32 * max(1.0, frame.shape[0] / 480.0):
                        label_slot += 1
                    else:
                        label_slot = 0
                    previous_marker_position = (float(predicted_frame_x), float(predicted_frame_y))
                    self.visualizer.draw_prediction_point(
                        frame, (predicted_frame_x, predicted_frame_y), f"+{horizon:.1f}s", label_slot
                    )

                    self.predictions_log.write(
                        [
                            self.frame_id,
                            current_time,
                            horizon,
                            predicted[0],
                            predicted[1],
                            velocity[0],
                            velocity[1],
                            acceleration[0],
                            acceleration[1],
                        ]
                    )

                    self.verifier.register(
                        self.frame_id, current_time, horizon, predicted, camera_scale
                    )

        trail = self.history.recent_points()
        if frame_from_world is not None and len(trail) > 0:
            history_frame_x, history_frame_y = to_frame(
                [point[0] for point in trail], [point[1] for point in trail]
            )
            trail = list(zip(history_frame_x, history_frame_y))
        self.visualizer.draw_history(frame, trail)
        self.visualizer.draw_hud(frame, current_time, self.verifier)
        if camera_state is not None:
            self.visualizer.draw_status(
                frame,
                (
                    "CAMERA: background (relative to background)"
                    if camera_state.reliable
                    else "CAMERA: no background - frozen"
                ),
                ok=camera_state.reliable,
            )

    def run(self, display: bool = True):
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
            pass
        self._write_metrics()

    def _write_metrics(self):
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

            print(
                f"{horizon:.1f}s | N={n:4d} | MAE={mae_out:.2f}px | RMSE={rmse_out:.2f}px"
            )

        metrics_log.close()
