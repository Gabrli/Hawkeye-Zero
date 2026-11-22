import cv2

from engine.utils import subtle


class Visualizer:
    """Draw detections, observed paths, predicted paths, and status text."""

    def __init__(
        self,
        prediction_color=(0, 255, 255),
        hud_alpha=1.0,
        compact_hud=False,
        style="bold",
    ):
        # Prediction color in OpenCV BGR order.
        self.prediction_color = tuple(int(c) for c in prediction_color)
        self._hud_bottom = 145
        # Values below 1.0 make the HUD background translucent.
        self.hud_alpha = float(hud_alpha)
        # Compact mode uses a small text column instead of a large metrics panel.
        self.compact_hud = bool(compact_hud)
        self._compact_y = 0
        # Supported styles: bold or subtle.
        self.style = style
        self._placed = (
            []
        )  # Rectangles already occupied by labels and prediction markers in this frame.

    def _dark_rect(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if self.hud_alpha >= 1.0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        else:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = (roi * (1.0 - self.hud_alpha)).astype(roi.dtype)

    @staticmethod
    def _ui(frame):
        # Scale drawing dimensions relative to a 480p frame.
        return max(1.0, frame.shape[0] / 480.0)

    def draw_detection(self, frame, x1, y1, x2, y2, cx, cy):
        ui = self._ui(frame)
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            max(2, round(2 * ui)),
        )
        cv2.circle(frame, (int(cx), int(cy)), max(5, round(5 * ui)), (0, 0, 255), -1)

    def draw_history(self, frame, points):
        for i in range(1, len(points)):
            p1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            p2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), max(2, round(2 * self._ui(frame))))

    def draw_prediction_curve(self, frame, x_pred, y_pred):
        ui = self._ui(frame)
        self._placed = []  # Reset label placement for the new frame.
        points = [(int(x_pred[i]), int(y_pred[i])) for i in range(len(x_pred))]
        if self.style == "subtle":
            subtle.draw_path(frame, points, self.prediction_color)
            return
        for i in range(
            len(points) - 1
        ):  # A dark outline keeps the prediction visible on bright and dark backgrounds.
            cv2.line(frame, points[i], points[i + 1], (0, 0, 0), max(4, round(4 * ui)))
        for i in range(len(points) - 1):
            cv2.line(
                frame,
                points[i],
                points[i + 1],
                self.prediction_color,
                max(2, round(2 * ui)),
            )

    def draw_prediction_point(self, frame, position, label, slot=0):
        # The slot argument is retained for compatibility; labels use collision checks.
        ui = self._ui(frame)
        x, y = int(position[0]), int(position[1])
        r = round(7 * ui)
        scale, th_in, th_out = 0.5 * ui, max(2, round(2 * ui)), max(4, round(4 * ui))
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th_in)
        pad, rr = round(3 * ui), round(10 * ui)

        def overlaps(a, b):
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

        cands = []
        for k in (0, 1, -1, 2, -2, 3, -3):
            dy = k * (tht + pad)
            cands.append((x + rr + pad, y + tht // 2 + dy))  # po prawej
            cands.append((x - rr - pad - tw, y + tht // 2 + dy))  # po lewej
        lx, ly = cands[0]
        for cx_, cy_ in cands:
            rect = (cx_, cy_ - tht, cx_ + tw, cy_ + base)
            if not any(overlaps(rect, q) for q in self._placed):
                lx, ly = cx_, cy_
                break

        if self.style == "subtle":
            subtle.draw_ring(frame, (x, y), self.prediction_color, radius=4.0)
            subtle.draw_label(frame, label, (lx, ly), self.prediction_color)
        else:
            cv2.circle(frame, (x, y), r, self.prediction_color, -1)
            cv2.circle(frame, (x, y), rr, (255, 255, 255), max(1, round(ui)))
            cv2.putText(
                frame,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                th_out,
            )
            cv2.putText(
                frame,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                th_in,
            )
        self._placed.append((lx, ly - tht, lx + tw, ly + base))
        self._placed.append((x - rr, y - rr, x + rr, y + rr))

    def _tiny(self, frame, text, color=(255, 255, 255)):
        # Scale text with frame height.
        ui = max(1.0, frame.shape[0] / 480.0)
        th = max(1, int(round(ui)))
        x = int(round(4 * ui))
        cv2.putText(
            frame,
            text,
            (x, self._compact_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3 * ui,
            (0, 0, 0),
            2 * th,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, self._compact_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3 * ui,
            color,
            th,
            cv2.LINE_AA,
        )
        self._compact_y += int(round(10 * ui))

    def draw_hud(self, frame, current_time, verifier):
        if self.compact_hud:
            self._compact_y = int(round(10 * max(1.0, frame.shape[0] / 480.0)))
            self._tiny(frame, f"{current_time:.1f}s")
            for horizon in verifier.horizons:
                mae = verifier.mae(horizon)
                self._tiny(
                    frame,
                    f"{horizon:g}s {mae:.1f}" if mae is not None else f"{horizon:g}s -",
                )
            return
        self._hud_bottom = 40 + 22 * len(
            verifier.horizons
        )  # Reserve enough HUD height for every configured horizon.
        self._dark_rect(frame, 5, 5, 400, self._hud_bottom)
        cv2.putText(
            frame,
            f"TIME: {current_time:.2f}s",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
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
                frame,
                text,
                (15, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            y_text += 22

    def draw_status(self, frame, text, ok=True):
        color = (0, 200, 0) if ok else (0, 165, 255)
        if self.compact_hud:
            self._tiny(frame, "bg OK" if ok else "bg --", color)
            return
        top = self._hud_bottom + 5
        self._dark_rect(frame, 5, top, 400, top + 25)
        cv2.putText(
            frame, text, (15, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
