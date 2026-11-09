from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraState:
    """Estimated camera transform and confidence information for one frame.

    Attributes:
        W: Homogeneous transform from current-frame coordinates to world coordinates.
        reliable: Whether the transform was updated from a reliable motion estimate.
        cut: Whether this frame starts a new shot.
        inliers: Number of inlier feature correspondences in the motion estimate.
        scale: Scale of ``W`` in world pixels per frame pixel.
    """

    W: np.ndarray          # 3x3, frame -> wordl
    reliable: bool       
    cut: bool              # is this the start of a new shot.
    inliers: int
    scale: float           # scale of W (world pixels per frame pixel)


class CameraMotion:
    """Estimate camera motion and map frame coordinates into a stable space."""

    def __init__(self, work_scale=0.5, min_inliers=25, min_ratio=0.5, cut_thresh=40.0,
                 max_corners=500, ransac_thresh=1.5, bbox_margin=0.3, min_texture=400,
                 static_mask=None, bbox_pad=6):
        """Initialize motion-estimation thresholds and reset tracking state.

        Args:
            work_scale: Resize factor used for optical-flow processing.
            min_inliers: Minimum affine-transform inliers for a reliable estimate.
            min_ratio: Minimum fraction of good matches that must be inliers.
            cut_thresh: Mean thumbnail difference that signals a shot change.
            max_corners: Maximum number of features to track.
            ransac_thresh: RANSAC reprojection threshold in working-image pixels.
            bbox_margin: Fractional padding around the excluded object bounding box.
            min_texture: Minimum number of textured pixels required for estimation.
            static_mask: Optional image mask of regions excluded from feature detection.
            bbox_pad: Additional padding in working-image pixels around the box.
        """
        self.work_scale = work_scale
        self.min_inliers = min_inliers
        self.min_ratio = min_ratio
        self.cut_thresh = cut_thresh
        self.max_corners = max_corners
        self.ransac_thresh = ransac_thresh
        self.bbox_margin = bbox_margin
        self.min_texture = min_texture
        self.static_mask = static_mask
        self.bbox_pad = bbox_pad
        self.reset()

    def reset(self):
        """Reset the accumulated transform and previous-frame image state."""
        self.W = np.eye(3)
        self._prev_gray = None
        self._prev_thumb = None

    @staticmethod
    def _scale_of(W):
        """Return the area-derived scale of a homogeneous affine transform.

        Args:
            W: A 3-by-3 affine transformation matrix.

        Returns:
            The square root of the absolute determinant of its linear component.
        """
        return float(np.sqrt(abs(np.linalg.det(W[:2, :2]))))

    def _mask(self, gray, bbox):
        """Build a feature mask excluding borders, static overlays, and an object.

        Args:
            gray: Grayscale working-resolution frame.
            bbox: Optional object box in full-resolution frame coordinates.

        Returns:
            An 8-bit mask whose nonzero pixels may be used for feature detection.
        """
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        b = 8
        mask[:b, :] = 0; mask[-b:, :] = 0; mask[:, :b] = 0; mask[:, -b:] = 0
    
        cols = np.where(gray.max(axis=0) < 12)[0]
        rows = np.where(gray.max(axis=1) < 12)[0]
        for c in cols:
            mask[:, max(c - 4, 0):c + 5] = 0
        for r in rows:
            mask[max(r - 4, 0):r + 5, :] = 0
        if self.static_mask is not None:
            sm = cv2.resize(self.static_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[sm > 0] = 0
        if bbox is not None:
            s = self.work_scale
            x1, y1, x2, y2 = [v * s for v in bbox]
            mx, my = (x2 - x1) * self.bbox_margin + self.bbox_pad, (y2 - y1) * self.bbox_margin + self.bbox_pad
            mask[int(max(y1 - my, 0)):int(y2 + my), int(max(x1 - mx, 0)):int(x2 + mx)] = 0
        return mask

    def update(self, frame, exclude_bbox=None) -> CameraState:
        """Estimate this frame's camera transform relative to the stable scene.

        Args:
            frame: BGR image in full-resolution frame coordinates.
            exclude_bbox: Optional object bounding box ``(x1, y1, x2, y2)`` to
                exclude from feature tracking.

        Returns:
            The current transform, reliability flag, shot-cut flag, inlier
            count, and scale.
        """
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, None, fx=self.work_scale, fy=self.work_scale, interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(gray_full, (64, 36)).astype(np.float32)

        # black/dark frames: no camera motion can be measured, and inliers would come from compression noise / object edges -> unreliable
        if gray.mean() < 8:
            return CameraState(self.W.copy(), False, False, 0, self._scale_of(self.W))

        cut = False
        if self._prev_thumb is not None and np.abs(thumb - self._prev_thumb).mean() > self.cut_thresh:
            self.reset()
            cut = True
        self._prev_thumb = thumb

        if self._prev_gray is None:
            self._prev_gray = gray
            return CameraState(self.W.copy(), True, cut, 0, self._scale_of(self.W))

        reliable, inliers = False, 0
        mask = self._mask(self._prev_gray, exclude_bbox)

        blur = cv2.GaussianBlur(self._prev_gray, (0, 0), 1.5).astype(np.float32)
        grad = np.hypot(cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1))
        if int(((grad > 12) & (mask > 0)).sum()) < self.min_texture:
            self._prev_gray = gray
            return CameraState(self.W.copy(), False, cut, 0, self._scale_of(self.W))
        p0 = cv2.goodFeaturesToTrack(self._prev_gray, self.max_corners, 0.02, 8, mask=mask, blockSize=7)
        if p0 is not None and len(p0) >= 30:
            lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **lk)
            fb = np.linalg.norm(p0 - p0r, axis=2).ravel()
            good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if good.sum() >= 30:
                M, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                                     ransacReprojThreshold=self.ransac_thresh, maxIters=2000)
                if M is not None:
                    inliers = int(inl.sum())
                    if inliers >= self.min_inliers and inliers / good.sum() >= self.min_ratio:
                        reliable = True
                        S = np.diag([self.work_scale, self.work_scale, 1.0])
                        M3 = np.vstack([M, [0, 0, 1]])
                        M_full = np.linalg.inv(S) @ M3 @ S        # previous frame -> current frame (full transformation)
                        self.W = self.W @ np.linalg.inv(M_full)   # current frame -> world

        self._prev_gray = gray
        return CameraState(self.W.copy(), reliable, cut, inliers, self._scale_of(self.W))


def build_static_hud_mask(video_path, samples=240, thresh=10, dilate=7):
    """Build a mask of visually static overlays in a video.

    Args:
        video_path: Path to the video to sample.
        samples: Number of evenly spaced frames to inspect.
        thresh: High-pass intensity threshold for marking overlay pixels.
        dilate: Width and height of the square mask-dilation kernel.

    Returns:
        A dilated 8-bit mask of static high-frequency regions.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    med = np.median(np.stack(frames), axis=0).astype(np.float32)
    hp = med - cv2.GaussianBlur(med, (0, 0), 4)
    mask = (np.abs(hp) > thresh).astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
