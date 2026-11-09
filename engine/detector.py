from ultralytics import YOLO
from engine.constants import MODEL_PATH, OBJECT_CLASS


class Detection:
    """Bounding box, center point, and confidence for one detected object."""

    __slots__ = ("cx", "cy", "x1", "y1", "x2", "y2", "confidence")

    def __init__(self, cx, cy, x1, y1, x2, y2, confidence):
        """Store the detection geometry and confidence.

        Args:
            cx: Horizontal coordinate of the box center.
            cy: Vertical coordinate of the box center.
            x1: Left edge of the bounding box.
            y1: Top edge of the bounding box.
            x2: Right edge of the bounding box.
            y2: Bottom edge of the bounding box.
            confidence: Model confidence for the detection.
        """
        self.cx = cx
        self.cy = cy
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence


class Yolo:
    """Detect the highest-confidence object of a selected YOLO class."""

    def __init__(self, model_path: str = MODEL_PATH, object_class: int = OBJECT_CLASS):
        """Load model weights and select the target class.

        Args:
            model_path: Path or Ultralytics identifier for model weights.
            object_class: Class index to retain from model predictions.
        """
        self.model = YOLO(model_path)
        self.object_class = object_class

    def detect(self, frame, conf: float, verbose: bool = False):
        """Return the best matching detection in a frame, if any.

        Args:
            frame: Image passed to the Ultralytics model.
            conf: Minimum confidence threshold for model detections.
            verbose: Whether Ultralytics should emit inference details.

        Returns:
            The highest-confidence matching :class:`Detection`, or ``None`` if
            no detection matches the configured class.
        """
        results = self.model(frame, verbose=verbose, conf=conf)

        best = None
        best_confidence = -1.0

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                if cls != self.object_class:
                    continue

                confidence = float(box.conf[0])

                if confidence <= best_confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                best = Detection(cx, cy, x1, y1, x2, y2, confidence)
                best_confidence = confidence

        return best