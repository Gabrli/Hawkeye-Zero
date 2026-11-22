
MODEL_PATH = "yolo11n.pt"
OBJECT_CLASS = 0


class Detection:
    __slots__ = ("cx", "cy", "x1", "y1", "x2", "y2", "confidence")

    def __init__(self, cx, cy, x1, y1, x2, y2, confidence):
        self.cx = cx
        self.cy = cy
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence


class Yolo:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        object_class: int = OBJECT_CLASS,
        imgsz: int = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("YOLO detection requires the 'ultralytics' package.") from error
        self.model = YOLO(model_path)
        self.object_class = object_class
        self.imgsz = imgsz  # None = use the Ultralytics default image size (640)

    def detect(self, frame, conf: float, verbose: bool = False):
        kwargs = {} if self.imgsz is None else {"imgsz": self.imgsz}
        results = self.model(frame, verbose=verbose, conf=conf, **kwargs)

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
