"""Face detection, alignment and cropping used by ``scripts/prepare_faces.py``.

A detector returns one face per frame as a box plus the two eye centres (YuNet via OpenCV's
``FaceDetectorYN``). The crop is rotated so that the eyes are horizontal, expanded by a margin and
resized to 224 x 224, which matches the preprocessing in the manuscript ("detected, aligned, and
cropped to the face region").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Face:
    box: tuple              # x, y, w, h in pixels
    right_eye: tuple        # (x, y) in the image, subject's right eye (left in the picture)
    left_eye: tuple
    score: float = 1.0


def align_crop(img: np.ndarray, face: Face, size: int = 224, margin: float = 1.3) -> np.ndarray:
    """Roll-align by the eye line, crop a square of ``margin`` x the face size, resize to ``size``."""
    x, y, w, h = face.box
    (rx, ry), (lx, ly) = face.right_eye, face.left_eye
    angle = np.degrees(np.arctan2(ly - ry, lx - rx))
    cx, cy = x + w / 2.0, y + h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, rot, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    side = int(round(max(w, h) * margin))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = side
    padded = cv2.copyMakeBorder(rotated, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class YuNetDetector:
    """OpenCV YuNet face detector (ONNX from the OpenCV model zoo, `face_detection_yunet_*.onnx`)."""

    def __init__(self, model_path: str, score_threshold: float = 0.6):
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)

    def __call__(self, bgr: np.ndarray) -> Optional[Face]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = faces[np.argmax(faces[:, -1] * faces[:, 2] * faces[:, 3])]   # most confident large face
        return Face(tuple(f[0:4]), (f[4], f[5]), (f[6], f[7]), float(f[-1]))
