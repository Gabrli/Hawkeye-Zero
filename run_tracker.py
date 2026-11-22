import argparse
import math
from pathlib import Path

from engine import Tracker
from engine.constants import HORIZONS, OBJECT_CLASS

COLORS = {
    "red": (25, 25, 180),
    "green": (0, 170, 0),
    "yellow": (0, 255, 255),
}


def _confidence(value):
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return confidence


def main(argv=None):
    """Parse command-line options and run the video tracker."""
    parser = argparse.ArgumentParser(description="Track an object and predict its trajectory in a video.")
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"), help="Ultralytics model weights")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="directory for output files")
    parser.add_argument("--class-id", type=int, default=OBJECT_CLASS, help="YOLO object class ID to track")
    parser.add_argument("--confidence", type=_confidence, default=0.35, help="minimum detection confidence (0-1)")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=HORIZONS,
        metavar="SECONDS",
        help="future prediction horizons in seconds (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true", help="show the annotated video while processing")
    parser.add_argument("--color", choices=COLORS, default="red", help="prediction overlay color")
    parser.add_argument("--camera-compensation", action="store_true", help="compensate for camera movement")
    args = parser.parse_args(argv)

    if args.class_id < 0:
        parser.error("--class-id must be non-negative")
    if any(not math.isfinite(horizon) or horizon <= 0 for horizon in args.horizons):
        parser.error("--horizons values must be finite and greater than zero")
    if not args.video.is_file():
        parser.error(f"video file does not exist: {args.video}")
    if args.weights != Path("yolo11n.pt") and not args.weights.is_file():
        parser.error(f"weights file does not exist: {args.weights}")

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
        object_class=args.class_id,
        confidence=args.confidence,
        horizons=args.horizons,
        prediction_color=COLORS[args.color],
        hud_alpha=0.6,
        compact_hud=True,
        camera_compensation=args.camera_compensation,
    )
    tracker.run(display=args.show)
    print(f"Test complete. Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
