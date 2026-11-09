import argparse
from pathlib import Path

from engine import Tracker
from engine.constants import HORIZONS, OBJECT_CLASS

COLORS = {
    "red": (25, 25, 180),
    "green": (0, 170, 0),
    "yellow": (0, 255, 255),
}


def main():
    """Parse command-line options and run the video tracker."""
    parser = argparse.ArgumentParser(description="Track an object and predict its trajectory in a video.")
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"), help="Ultralytics model weights")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="directory for output files")
    parser.add_argument("--show", action="store_true", help="show the annotated video while processing")
    parser.add_argument("--color", choices=COLORS, default="red", help="prediction overlay color")
    parser.add_argument("--camera-compensation", action="store_true", help="compensate for camera movement")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(
        str(args.video),
        output_video=str(args.output_dir / "output.mp4"),
        output_detections=str(args.output_dir / "detections.csv"),
        output_predictions=str(args.output_dir / "predictions.csv"),
        output_comparison=str(args.output_dir / "trajectory_comparison.csv"),
        output_metrics=str(args.output_dir / "metrics.csv"),
        output_camera=str(args.output_dir / "camera.csv"),
        model_path=str(args.weights),
        object_class=OBJECT_CLASS,
        confidence=0.35,
        horizons=HORIZONS,
        prediction_color=COLORS[args.color],
        hud_alpha=0.6,
        compact_hud=True,
        camera_compensation=args.camera_compensation,
    )
    tracker.run(display=args.show)


if __name__ == "__main__":
    main()
