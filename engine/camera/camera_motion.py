"""Estimate frame-to-frame camera motion from background optical flow.

The cumulative transform maps current-frame points into stabilized coordinates.
Per-frame ``step`` transforms map previous-frame points into current-frame points.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from engine.camera import (
    WORK_SCALE, BBOX_MARGIN, BBOX_PAD, MIN_TEXTURE, MAX_CORNERS, MIN_CORNERS,
    RANSAC_THRESH, CUT_THRESH, MIN_INLIERS, MIN_RATIO,
)

@dataclass
class CameraState:
    """Camera-motion result for one frame.

    ``world_from_frame`` maps current-frame points into stabilized coordinates.
    ``previous_to_current`` maps previous-frame points into current-frame points.
    """

    world_from_frame: np.ndarray
    reliable: bool
    cut: bool
    inliers: int
    scale: float
    previous_to_current: np.ndarray | None = None

    def __post_init__(self):
        if self.previous_to_current is None:
            self.previous_to_current = np.eye(3)

    @property
    def W(self):
        """Backward-compatible alias for ``world_from_frame``."""
        return self.world_from_frame

    @property
    def step(self):
        """Backward-compatible alias for ``previous_to_current``."""
        return self.previous_to_current


class CameraMotion:
    def __init__(
        self,
        work_scale=WORK_SCALE,
        min_inliers=MIN_INLIERS,
        min_ratio=MIN_RATIO,
        cut_thresh=CUT_THRESH,
        max_corners=MAX_CORNERS,
        ransac_thresh=RANSAC_THRESH,
        bbox_margin=BBOX_MARGIN,
        min_texture=MIN_TEXTURE,
        static_mask=None,
        bbox_pad=BBOX_PAD,
    ):
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

    @property
    def W(self):
        """Backward-compatible alias for ``world_from_frame``."""
        return self.world_from_frame

    @W.setter
    def W(self, transform):
        self.world_from_frame = transform

    def reset(self):
        self.world_from_frame = np.eye(3)
        self._prev_gray = None
        self._prev_thumb = None

    @staticmethod
    def _scale_of(transform):
        # The square root of the linear-part determinant gives the 2D scale.
        return float(np.sqrt(abs(np.linalg.det(transform[:2, :2]))))

    def _mask(self, gray, bbox):
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        b = 8
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
        # black borders: optical flow is unreliable at the edges, and the camera motion is not defined there
        cols = np.where(gray.max(axis=0) < 12)[0]
        rows = np.where(gray.max(axis=1) < 12)[0]
        for c in cols:
            mask[:, max(c - 4, 0) : c + 5] = 0
        for r in rows:
            mask[max(r - 4, 0) : r + 5, :] = 0
        if self.static_mask is not None:
            sm = cv2.resize(self.static_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask[sm > 0] = 0
        if bbox is not None:
            s = self.work_scale
            x1, y1, x2, y2 = [v * s for v in bbox]
            mx, my = (
                (x2 - x1) * self.bbox_margin + self.bbox_pad,
                (y2 - y1) * self.bbox_margin + self.bbox_pad,
            )
            mask[
                int(max(y1 - my, 0)) : int(y2 + my), int(max(x1 - mx, 0)) : int(x2 + mx)
            ] = 0
        return mask

    def update(self, frame, exclude_bbox=None) -> CameraState:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray_full,
            None,
            fx=self.work_scale,
            fy=self.work_scale,
            interpolation=cv2.INTER_AREA,
        )
        thumb = cv2.resize(gray_full, (64, 36)).astype(np.float32)

        # black/dark frames: no camera motion can be measured, and inliers would come from compression noise / object edges -> not reliable
        if gray.mean() < 8:
            return CameraState(
                self.world_from_frame.copy(),
                False,
                False,
                0,
                self._scale_of(self.world_from_frame),
            )

        cut = False
        if (
            self._prev_thumb is not None
            and np.abs(thumb - self._prev_thumb).mean() > self.cut_thresh
        ):
            self.reset()
            cut = True
        self._prev_thumb = thumb

        if self._prev_gray is None:
            self._prev_gray = gray
            return CameraState(
                self.world_from_frame.copy(),
                True,
                cut,
                0,
                self._scale_of(self.world_from_frame),
            )

        reliable, inliers = False, 0
        step = np.eye(3)
        mask = self._mask(self._prev_gray, exclude_bbox)
        # observation gate - if the previous frame has too little texture, the camera motion cannot be measured, and the inliers would come from compression noise / object edges -> not reliable

        blur = cv2.GaussianBlur(self._prev_gray, (0, 0), 1.5).astype(np.float32)
        grad = np.hypot(
            cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
        )
        if int(((grad > 12) & (mask > 0)).sum()) < self.min_texture:
            self._prev_gray = gray
            return CameraState(
                self.world_from_frame.copy(),
                False,
                cut,
                0,
                self._scale_of(self.world_from_frame),
            )
        # get good features to track in the previous frame, and track them to the current frame using optical flow
        previous_points = cv2.goodFeaturesToTrack(
            self._prev_gray, self.max_corners, 0.02, 8, mask=mask, blockSize=7
        )
        if previous_points is not None and len(previous_points) >= MIN_CORNERS:
            lk = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            # Lucas-Kanade - follow the features from the previous frame to the current frame, and back to the previous frame, to filter out bad matches
            current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, previous_points, None, **lk
            )
            if current_points is not None and forward_status is not None:
                returned_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self._prev_gray, current_points, None, **lk
                )
            else:
                returned_points, backward_status = None, None

            if returned_points is not None and backward_status is not None:
                # Reject tracks that do not return close to their starting point.
                forward_backward_error = np.linalg.norm(
                    previous_points - returned_points, axis=2
                ).ravel()
                valid_tracks = (
                    (forward_status.ravel() == 1)
                    & (backward_status.ravel() == 1)
                    & (forward_backward_error < 1.0)
                )
            else:
                valid_tracks = np.zeros(len(previous_points), dtype=bool)

            if valid_tracks.sum() >= MIN_CORNERS:
                # Fit the previous-to-current camera transform and reject outliers with RANSAC.
                affine_transform, inlier_flags = cv2.estimateAffinePartial2D(
                    previous_points[valid_tracks],
                    current_points[valid_tracks],
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_thresh,
                    maxIters=2000,
                )
                if affine_transform is not None and inlier_flags is not None:
                    inliers = int(inlier_flags.sum())
                    if (
                        inliers >= self.min_inliers
                        and inliers / valid_tracks.sum() >= self.min_ratio
                    ):
                        reliable = True
                        scale_matrix = np.diag([self.work_scale, self.work_scale, 1.0])
                        affine_homogeneous = np.vstack([affine_transform, [0, 0, 1]])
                        full_resolution_step = (
                            np.linalg.inv(scale_matrix)
                            @ affine_homogeneous
                            @ scale_matrix
                        )
                        # Compose the inverse frame step into the current-to-stabilized transform.
                        self.world_from_frame = self.world_from_frame @ np.linalg.inv(full_resolution_step)
                        step = full_resolution_step

        self._prev_gray = gray
        return CameraState(
            world_from_frame=self.world_from_frame.copy(),
            reliable=reliable,
            cut=cut,
            inliers=inliers,
            scale=self._scale_of(self.world_from_frame),
            previous_to_current=step,
        )


def build_static_hud_mask(video_path, samples=240, thresh=10, dilate=7, bottom_px=0):
    """Build a mask for persistent high-frequency HUD elements in a video."""
    if samples < 1 or dilate < 1:
        raise ValueError("samples and dilate must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError("No frames could be sampled to build the HUD mask")
    med = np.median(np.stack(frames), axis=0).astype(np.float32)
    hp = med - cv2.GaussianBlur(med, (0, 0), 4)
    mask = (np.abs(hp) > thresh).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    if bottom_px > 0:
        mask[-bottom_px:, :] = 255
    return mask
