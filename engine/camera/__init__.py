"""Camera, image-processing, and terrain-analysis components."""

# Configuration is declared before importing submodules to avoid circular imports.
CLASS_NAMES = [
    "smooth regolith",
    "rough terrain",
    "rocks & gravel",
    "shadow / depression",
]
CLASS_COLORS = [(190, 140, 70), (70, 170, 90), (60, 225, 255), (170, 70, 160)]  # BGR
WORK_SCALE = 0.5
HF_T = 2.4
BLOB_T = 13.0
DARK_T = -7.0
ROCK_MIN = 30
ROCK_MAX = 1500
MAX_ASPECT = 2.6
MODE_K = 15
CUT_THRESH = 40.0
MIN_TEXTURE = 400
BBOX_MARGIN = 0.3
BBOX_PAD = 6
RANSAC_THRESH = 1.5
MAX_CORNERS = 500
MIN_CORNERS = 30
MIN_INLIERS = 25
MIN_RATIO = 0.5

from engine.camera.detector import Detection, Yolo
from engine.camera.segmentation import TerrainSegmenter

__all__ = [
    "Detection", "Yolo", "TerrainSegmenter", "CLASS_NAMES", "CLASS_COLORS",
    "WORK_SCALE", "HF_T", "BLOB_T", "DARK_T", "ROCK_MIN", "ROCK_MAX",
    "MAX_ASPECT", "MODE_K", "CUT_THRESH", "MIN_TEXTURE", "BBOX_MARGIN",
    "BBOX_PAD", "RANSAC_THRESH", "MAX_CORNERS", "MIN_CORNERS",
    "MIN_INLIERS", "MIN_RATIO",
]
