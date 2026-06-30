"""
Image preprocessing module.

Performs:
  - BGR -> RGB color space conversion
  - Image resize to model input dimensions (letterbox)
  - Normalization (0..255 -> 0..1)
  - HWC -> CHW layout conversion
  - Batch dimension expansion

Matches the Ultralytics preprocessing pipeline for consistent results.
"""

import cv2
import numpy as np
from typing import Tuple


class Preprocessor:
    def __init__(
        self,
        input_width: int = 640,
        input_height: int = 640,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        half: bool = False,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.half = half
        self.dtype = np.float16 if half else np.float32

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Preprocess a BGR frame for YOLO inference.

        Args:
            frame: numpy array (H, W, 3) in BGR format

        Returns:
            tensor:   (1, 3, H, W) float32/float16 normalized tensor
            ratio:    (ratio_w, ratio_h) scaling factors for coordinate mapping
            pad:      (pad_w, pad_h) padding offsets for letterbox
        """
        # Letterbox resize with padding to preserve aspect ratio
        img, ratio, pad_info = self._letterbox(frame)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, normalize to [0, 1]
        img = img.transpose(2, 0, 1).astype(self.dtype) / 255.0

        # Apply mean/std normalization (typically identity for YOLO)
        img = (img - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)

        # Add batch dimension
        tensor = np.expand_dims(img, axis=0)

        return tensor, ratio, pad_info

    def _letterbox(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Resize and pad image to fit model input size while preserving aspect ratio.

        Returns:
            img:    padded image of exactly (input_height, input_width, 3)
            ratio:  (ratio_w, ratio_h) scale factors
            pad:    (pad_left, pad_top) padding offsets
        """
        h0, w0 = frame.shape[:2]
        r_h = self.input_height / h0
        r_w = self.input_width / w0
        ratio = min(r_h, r_w)

        new_w = int(round(w0 * ratio))
        new_h = int(round(h0 * ratio))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = self.input_width - new_w
        dh = self.input_height - new_h

        pad_left = dw // 2
        pad_top = dh // 2
        pad_right = dw - pad_left
        pad_bottom = dh - pad_top

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        ratio = (ratio, ratio)
        pad_info = (pad_left, pad_top)

        return padded, ratio, pad_info


class PreprocessorSimple:
    """Simplified preprocessor: direct resize (no letterbox), BGR->RGB, CHW, /255."""

    def __init__(self, input_width: int = 640, input_height: int = 640):
        self.input_width = input_width
        self.input_height = input_height

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)
