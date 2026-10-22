"""
Post-processing module for YOLOv11 ONNX outputs.

Handles:
  - Coordinate decoding (center-xywh -> xyxy)
  - Confidence filtering
  - Non-Maximum Suppression (NMS)
  - Box coordinate scaling back to original frame
  - Detection result rendering on BGR frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2) in original frame coordinates
    confidence: float
    class_id: int
    class_name: str


class PostProcessor:
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 300,
        class_names: Optional[List[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_names = class_names or [f"class_{i}" for i in range(100)]

        # Colors for visualization (BGR format)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        self.colors = self.colors.tolist()

    def __call__(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        ratio: Tuple[float, float],
        pad: Tuple[int, int],
    ) -> List[Detection]:
        """
        Process raw YOLO output into a list of Detection objects.

        Args:
            output:          Raw model output. Shape format:
                             (1, 84, N) - [x_center, y_center, w, h, class_probs...]
                             (1, N, 84) - transposed version
                             (1, 4+num_classes, N) - alternative layout
            original_shape:  (H, W) of the original input frame
            ratio:           (ratio_w, ratio_h) scaling factors from letterbox
            pad:             (pad_left, pad_top) padding offsets

        Returns:
            List of Detection objects filtered and NMS'd
        """
        # Adapt to different output layouts
        # YOLOv11 ONNX output: (1, 84, 8400) = [4 bbox coords + 80 class scores] x 8400 anchors
        # Some exports: (1, 8400, 84) - transposed

        output_data = output[0]  # Remove batch dimension -> (84, N) or (N, 84)

        # Detect layout: columns=84 means (N, 84), otherwise (84, N)
        if output_data.shape[1] == 84:
            output_data = output_data.T  # (84, N)

        predictions = output_data.T  # (N, 84) where 84 = 4 bbox + 80 class scores

        bbox_raw = predictions[:, :4]        # (N, 4) center_x, center_y, w, h
        class_scores = predictions[:, 4:]    # (N, num_classes)

        # Get max class score and class ID per anchor
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Filter by confidence threshold
        mask = max_scores > self.confidence_threshold
        if not np.any(mask):
            return []

        bbox_raw = bbox_raw[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Decode center-xywh to xyxy in model input space
        xyxy = self._decode_boxes(bbox_raw)

        # Scale boxes from model input space to original frame space
        xyxy_original = self._scale_boxes(
            xyxy,
            model_shape=(640, 640),
            original_shape=original_shape,
            ratio=ratio,
            pad=pad,
        )

        # NMS per class
        detections = self._nms(xyxy_original, max_scores, class_ids)

        return detections

    def _decode_boxes(self, bbox_raw: np.ndarray) -> np.ndarray:
        """
        Decode center-xywh format to xyxy (top-left, bottom-right).

        YOLOv11 raw output: (cx, cy, w, h) as fraction of model input size.
        Returns: (x1, y1, x2, y2) in pixel coordinates of the model input.
        """
        xyxy = np.zeros_like(bbox_raw)
        xyxy[:, 0] = bbox_raw[:, 0] - bbox_raw[:, 2] / 2.0  # x1 = cx - w/2
        xyxy[:, 1] = bbox_raw[:, 1] - bbox_raw[:, 3] / 2.0  # y1 = cy - h/2
        xyxy[:, 2] = bbox_raw[:, 0] + bbox_raw[:, 2] / 2.0  # x2 = cx + w/2
        xyxy[:, 3] = bbox_raw[:, 1] + bbox_raw[:, 3] / 2.0  # y2 = cy + h/2
        return xyxy

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        model_shape: Tuple[int, int],
        original_shape: Tuple[int, int],
        ratio: Tuple[float, float],
        pad: Tuple[int, int],
    ) -> np.ndarray:
        """
        Scale boxes from model input coordinates back to original frame coordinates.

        Reverses the letterbox transformation:
          1. Remove padding
          2. Undo scaling
          3. Clip to image bounds
        """
        orig_h, orig_w = original_shape
        r_w, r_h = ratio
        pad_w, pad_h = pad

        boxes[:, 0] -= pad_w
        boxes[:, 1] -= pad_h
        boxes[:, 2] -= pad_w
        boxes[:, 3] -= pad_h

        boxes[:, 0] /= r_w
        boxes[:, 1] /= r_h
        boxes[:, 2] /= r_w
        boxes[:, 3] /= r_h

        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        return boxes

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
    ) -> List[Detection]:
        """Per-class Non-Maximum Suppression using OpenCV's built-in NMS."""
        detections = []

        # Group by class
        unique_classes = np.unique(class_ids)
        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            # OpenCV NMS expects boxes in (x1, y1, w, h) format
            boxes_wh = cls_boxes.copy()
            boxes_wh[:, 2] -= boxes_wh[:, 0]
            boxes_wh[:, 3] -= boxes_wh[:, 1]

            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_wh.tolist(),
                scores=cls_scores.tolist(),
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )

            if len(indices) == 0:
                continue

            for idx in indices.flatten():
                x1, y1, x2, y2 = cls_boxes[idx]
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(cls_scores[idx]),
                    class_id=int(cls_id),
                    class_name=self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else f"class_{cls_id}",
                ))

        # Sort by confidence descending, respect max_detections
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:self.max_detections]


class Renderer:
    """Draw detection results on BGR frames."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        draw_labels: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.draw_labels = draw_labels
        self.box_thickness = box_thickness
        self.font_scale = font_scale

        self.class_names = class_names
        if class_names:
            np.random.seed(42)
            self.colors = {
                i: tuple(int(c) for c in np.random.randint(0, 255, 3))
                for i in range(len(class_names))
            }
        else:
            self.colors = {}

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes and labels on the frame (in-place)."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.colors.get(det.class_id, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            if self.draw_labels:
                label = f"{det.class_name} {det.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        return frame


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Pure numpy NMS implementation (no OpenCV dependency).

    boxes: (N, 4) [x1, y1, x2, y2]
    scores: (N,)
    Returns: indices of kept boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-8)

        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int32)
