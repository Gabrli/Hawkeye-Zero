"""Backward-compatible tracker import path.

New code should import :class:`Tracker` from ``engine`` or
``engine.tracking.tracker``.
"""

import engine.tracking.tracker as _implementation
from engine.camera.detector import Yolo
from engine.tracking.tracker import Tracker as _Tracker

cv2 = _implementation.cv2


class Tracker(_Tracker):
    """Compatibility wrapper that preserves module-level detector patching."""

    def __init__(self, *args, **kwargs):
        _implementation.Yolo = Yolo
        super().__init__(*args, **kwargs)
