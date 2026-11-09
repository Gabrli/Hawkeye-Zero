import csv
from types import SimpleNamespace

import numpy as np

import engine.tracker as tracker_module
from engine.tracker import Tracker


class SyntheticCapture:
    def __init__(self, frames, fps):
        self.frames = iter(frames)
        self.fps = fps

    def isOpened(self):
        return True

    def get(self, property_id):
        return {
            tracker_module.cv2.CAP_PROP_FPS: self.fps,
            tracker_module.cv2.CAP_PROP_FRAME_WIDTH: 64,
            tracker_module.cv2.CAP_PROP_FRAME_HEIGHT: 48,
        }[property_id]

    def read(self):
        try:
            return True, next(self.frames).copy()
        except StopIteration:
            return False, None

    def release(self):
        pass


class SyntheticWriter:
    def __init__(self, *args):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        pass


class SyntheticDetector:
    def __init__(self, **kwargs):
        self.frame_index = 0

    def detect(self, frame, conf):
        center_x = 15.0 + 2.0 * self.frame_index
        self.frame_index += 1
        return SimpleNamespace(
            cx=center_x,
            cy=24.0,
            x1=center_x - 2,
            y1=22.0,
            x2=center_x + 2,
            y2=26.0,
            confidence=0.9,
        )


def test_tracker_run_writes_video_and_prediction_csvs(tmp_path, monkeypatch):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
    writer = SyntheticWriter()
    monkeypatch.setattr(
        tracker_module.cv2,
        "VideoCapture",
        lambda path: SyntheticCapture(frames, fps=10.0),
    )
    monkeypatch.setattr(tracker_module.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(tracker_module, "Yolo", SyntheticDetector)
    monkeypatch.setattr(tracker_module.cv2, "destroyAllWindows", lambda: None)

    tracker = Tracker(
        "synthetic.mp4",
        output_video=str(tmp_path / "output.mp4"),
        output_detections=str(tmp_path / "detections.csv"),
        output_predictions=str(tmp_path / "predictions.csv"),
        output_comparison=str(tmp_path / "comparison.csv"),
        output_metrics=str(tmp_path / "metrics.csv"),
        horizons=[0.1],
    )
    tracker.run(display=False)

    assert len(writer.frames) == 10
    with (tmp_path / "detections.csv").open(newline="") as file:
        assert len(list(csv.reader(file))) == 11
    with (tmp_path / "predictions.csv").open(newline="") as file:
        predictions = list(csv.reader(file))
    with (tmp_path / "comparison.csv").open(newline="") as file:
        comparisons = list(csv.reader(file))
    with (tmp_path / "metrics.csv").open(newline="") as file:
        metrics = list(csv.reader(file))

    assert len(predictions) > 1
    assert len(comparisons) > 1
    assert metrics[1][0] == "0.1"
    assert int(metrics[1][1]) > 0
