import cv2


class Visualizer:
    """Draw detections, trajectories, and status information onto video frames."""

    def __init__(self, prediction_color=(0, 255, 255), hud_alpha=1.0, compact_hud=False):
        """Configure prediction styling and HUD layout.

        Args:
            prediction_color: BGR color for predicted trajectories and points.
            hud_alpha: Darkness of the HUD background, from transparent to opaque.
            compact_hud: Whether to use the compact HUD layout.
        """
        self.prediction_color = tuple(int(c) for c in prediction_color)
        self._hud_bottom = 145
   
        self.hud_alpha = float(hud_alpha)
       
        self.compact_hud = bool(compact_hud)
        self._compact_y = 0

    def _dark_rect(self, frame, x1, y1, x2, y2):
        """Darken a clipped frame region to improve text contrast."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if self.hud_alpha >= 1.0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        else:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = (roi * (1.0 - self.hud_alpha)).astype(roi.dtype)

    def draw_detection(self, frame, x1, y1, x2, y2, cx, cy):
        """Draw a bounding box and center marker in place.

        Args:
            frame: BGR image to annotate.
            x1: Left edge of the detected box.
            y1: Top edge of the detected box.
            x2: Right edge of the detected box.
            y2: Bottom edge of the detected box.
            cx: Horizontal coordinate of the detection center.
            cy: Vertical coordinate of the detection center.
        """
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    def draw_history(self, frame, points):
        """Draw connected line segments through observed positions.

        Args:
            frame: BGR image to annotate.
            points: Ordered sequence of ``(x, y)`` positions.
        """
        for i in range(1, len(points)):
            p1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            p2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    def draw_prediction_curve(self, frame, x_pred, y_pred):
        """Draw an outlined predicted trajectory in place.

        Args:
            frame: BGR image to annotate.
            x_pred: Predicted horizontal coordinates.
            y_pred: Predicted vertical coordinates.
        """
        points = [(int(x_pred[i]), int(y_pred[i])) for i in range(len(x_pred))]
        for i in range(len(points) - 1):  # czarny obrys pod linią - czytelność na jasnym i ciemnym tle
            cv2.line(frame, points[i], points[i + 1], (0, 0, 0), 4)
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], self.prediction_color, 2)

    def draw_prediction_point(self, frame, position, label, slot=0):
        """Draw a predicted position marker and its label.

        Args:
            frame: BGR image to annotate.
            position: Predicted ``(x, y)`` frame coordinates.
            label: Text label shown beside the marker.
            slot: Additional vertical offset used to separate nearby labels.
        """
        x, y = int(position[0]), int(position[1])
        ly = y - 10 - 13 * slot
        cv2.circle(frame, (x, y), 7, self.prediction_color, -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 1)
        cv2.putText(
            frame, label, (x + 10, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4,
        )
        cv2.putText(
            frame, label, (x + 10, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    def _tiny(self, frame, text, color=(255, 255, 255)):
        """Draw one compact HUD line and advance the compact text cursor."""
        cv2.putText(frame, text, (4, self._compact_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (4, self._compact_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        self._compact_y += 10

    def draw_hud(self, frame, current_time, verifier):
        """Draw elapsed time and prediction error metrics.

        Args:
            frame: BGR image to annotate.
            current_time: Current video time in seconds.
            verifier: Prediction evaluator providing horizons and error metrics.
        """
        if self.compact_hud:
            self._compact_y = 10
            self._tiny(frame, f"{current_time:.1f}s")
            for horizon in verifier.horizons:
                mae = verifier.mae(horizon)
                self._tiny(frame, f"{horizon:g}s {mae:.1f}" if mae is not None else f"{horizon:g}s -")
            return
        self._hud_bottom = 40 + 22 * len(verifier.horizons) 
        self._dark_rect(frame, 5, 5, 400, self._hud_bottom)
        cv2.putText(
            frame, f"TIME: {current_time:.2f}s", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        y_text = 55
        for horizon in verifier.horizons:
            mae = verifier.mae(horizon)
            rmse = verifier.rmse(horizon)

            if mae is not None:
                text = f"{horizon:.1f}s MAE:{mae:.1f} RMSE:{rmse:.1f}"
            else:
                text = f"{horizon:.1f}s MAE:--- RMSE:---"

            cv2.putText(
                frame, text, (15, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
            )
            y_text += 22

    def draw_status(self, frame, text, ok=True):
        """Draw a status message using a color indicating success or warning.

        Args:
            frame: BGR image to annotate.
            text: Status text for the full HUD layout.
            ok: Whether the status should be styled as successful.
        """
        color = (0, 200, 0) if ok else (0, 165, 255)
        if self.compact_hud:
            self._tiny(frame, "bg OK" if ok else "bg --", color)
            return
        top = self._hud_bottom + 5
        self._dark_rect(frame, 5, top, 400, top + 25)
        cv2.putText(frame, text, (15, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
