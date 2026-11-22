"""Heuristic terrain segmentation for lunar surface images."""

import cv2
import numpy as np
from engine.camera import (
    BLOB_T,
    CLASS_COLORS,
    CLASS_NAMES,
    DARK_T,
    HF_T,
    MAX_ASPECT,
    MODE_K,
    ROCK_MAX,
    ROCK_MIN,
    WORK_SCALE,
)


class TerrainSegmenter:
    """Segment image regions using hand-crafted texture and contrast features."""

    def __init__(
        self,
        class_names=CLASS_NAMES,
        class_colors=CLASS_COLORS,
        work_scale=WORK_SCALE,
        hf_t=HF_T,
        blob_t=BLOB_T,
        dark_t=DARK_T,
        rock_min=ROCK_MIN,
        rock_max=ROCK_MAX,
        max_aspect=MAX_ASPECT,
        mode_k=MODE_K,
    ):
        if not 0 < work_scale <= 1:
            raise ValueError("work_scale must be greater than 0 and at most 1")
        if mode_k < 1:
            raise ValueError("mode_k must be positive")
        self.work_scale = work_scale
        self.hf_t, self.blob_t, self.dark_t = hf_t, blob_t, dark_t
        self.rock_min, self.rock_max, self.max_aspect = rock_min, rock_max, max_aspect
        self.mode_k = mode_k
        self.class_names = class_names
        self.class_colors = class_colors

    def features(self, bgr_frame):
        """Return texture, bright-blob, and dark-region feature maps."""
        resized_frame = cv2.resize(
            bgr_frame,
            None,
            fx=self.work_scale,
            fy=self.work_scale,
            interpolation=cv2.INTER_AREA,
        )
        # Use Lab lightness to build local texture and contrast features.
        lightness = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        texture_response = cv2.GaussianBlur(
            cv2.GaussianBlur(
                np.abs(lightness - cv2.GaussianBlur(lightness, (0, 0), 2)),
                (0, 0),
                3,
            ),
            (0, 0),
            4,
        )
        bright_blob_response = cv2.GaussianBlur(
            lightness - cv2.GaussianBlur(lightness, (0, 0), 8), (0, 0), 1.5
        )
        # Blur subtraction highlights dark local regions such as shadows and depressions.
        dark_region_response = (
            cv2.GaussianBlur(lightness, (0, 0), 6)
            - cv2.GaussianBlur(lightness, (0, 0), 25)
        )
        return texture_response, bright_blob_response, dark_region_response

    def segment(self, bgr_frame):
        """Return class labels, candidate rock boxes, and per-class pixel fractions."""
        (
            texture_response,
            bright_blob_response,
            dark_region_response,
        ) = self.features(bgr_frame)
        labels = (texture_response >= self.hf_t).astype(np.uint8)
        labels[
            (dark_region_response < self.dark_t)
            & (texture_response < self.hf_t + 1.5)
        ] = 3

        # Clean bright candidates before grouping them into connected components.
        bright_candidate_mask = (bright_blob_response > self.blob_t).astype(np.uint8)
        cleaned_blob_mask = cv2.morphologyEx(
            bright_candidate_mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(cleaned_blob_mask)
        )
        elements = []
        keep = np.zeros_like(cleaned_blob_mask)
        original_resolution_scale = 1.0 / self.work_scale
        for component_id in range(1, component_count):
            x, y, width, height, area = component_stats[component_id]
            if not (self.rock_min <= area <= self.rock_max):
                # Reject components outside the configured area range.
                continue
            if (
                max(width, height) / max(min(width, height), 1) > self.max_aspect
            ):  # too elongated: probably a shadow, not a rock
                continue
            if (
                area / float(max(width * height, 1)) < 0.4
            ):  # too sparse: probably a shadow, not a rock
                continue
            keep[component_labels == component_id] = 1
            elements.append(
                (
                    int(x * original_resolution_scale),
                    int(y * original_resolution_scale),
                    int((x + width) * original_resolution_scale),
                    int((y + height) * original_resolution_scale),
                    int(area * original_resolution_scale * original_resolution_scale),
                )
            )
        labels[cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0] = 2

        if self.mode_k > 1:  # Remove small isolated regions of a different class.
            labels = cv2.medianBlur(labels, self.mode_k | 1)
        class_fractions = [float((labels == class_id).mean()) for class_id in range(4)]
        return labels, elements, class_fractions
