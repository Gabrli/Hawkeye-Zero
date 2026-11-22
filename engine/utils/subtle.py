import cv2
import numpy as np

SUBTLE_COLOR = (
    240,
    235,
    200,
)  # BGR


def ui(frame):
    return max(1.0, frame.shape[0] / 480.0)


def _dashed(img, p1, p2, color, thick, dash, gap):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1:
        return
    d = (p2 - p1) / length
    pos = 0.0
    while pos < length:
        a, b = p1 + d * pos, p1 + d * min(pos + dash, length)
        cv2.line(
            img,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            color,
            thick,
            cv2.LINE_AA,
        )
        pos += dash + gap


def _blend(frame, draw, alpha):
    overlay = frame.copy()
    draw(overlay)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_path(frame, pts, color=SUBTLE_COLOR, alpha=0.55):
    """Crow's foot path (dashed line) connecting a sequence of points."""
    if len(pts) < 2:
        return
    u = ui(frame)
    thick, dash, gap = (
        max(1, int(round(1.1 * u))),
        max(4, int(round(9 * u))),
        max(3, int(round(7 * u))),
    )

    def draw(img):
        for a, b in zip(pts[:-1], pts[1:]):
            _dashed(img, a, b, color, thick, dash, gap)

    _blend(frame, draw, alpha)


def draw_ring(frame, xy, color=SUBTLE_COLOR, radius=4.0, alpha=0.7, cross=False):
    """Empty circle (optionally with a cross - a marker for the point of contact)."""
    u = ui(frame)
    r, thick = max(2, int(round(radius * u))), max(1, int(round(u * 0.9)))
    x, y = int(round(xy[0])), int(round(xy[1]))

    def draw(img):
        cv2.circle(img, (x, y), r, color, thick, cv2.LINE_AA)
        if cross:
            t = int(round(r * 0.7))
            cv2.line(img, (x - r - t, y), (x - r + 1, y), color, thick, cv2.LINE_AA)
            cv2.line(img, (x + r - 1, y), (x + r + t, y), color, thick, cv2.LINE_AA)
            cv2.line(img, (x, y - r - t), (x, y - r + 1), color, thick, cv2.LINE_AA)
            cv2.line(img, (x, y + r - 1), (x, y + r + t), color, thick, cv2.LINE_AA)

    _blend(frame, draw, alpha)


def draw_label(frame, text, xy, color=SUBTLE_COLOR, scale=0.38):
    """Small label with a subtle shadow (without a thick outline)."""
    u = ui(frame)
    x, y = int(round(xy[0])), int(round(xy[1]))
    s, th = scale * u, max(1, int(round(u * 0.8)))
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        s,
        (20, 20, 20),
        th,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, th, cv2.LINE_AA
    )
