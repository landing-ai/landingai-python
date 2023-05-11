import logging
from typing import Any, Dict, List

import cv2
import numpy as np
from landingai.common import (
    Prediction,
    ObjectDetectionPrediction,
    SegmentationPrediction,
)
from segmentation_mask_overlay import overlay_masks
from PIL import Image

SPACING_PIXELS = -5


def overlay_predictions(
    predictions: List[Prediction], image: np.ndarray
) -> Image.Image:
    """Overlay the prediction results on the input image and return the image with overlaid."""
    types = {type(pred) for pred in predictions}
    assert len(types) == 1, f"Expecting only one type of prediction, got {types}"
    overlay_func = _OVERLAY_FUNC_MAP[types.pop()]
    return overlay_func(predictions, image)


def overlay_bboxes(predictions: List[ObjectDetectionPrediction], image: np.ndarray) -> Image.Image:
    "Draw bounding boxes on the input image and return the image with bounding boxes drawn."
    color = (255, 0, 0)
    thickness = 2
    for pred in predictions:
        bbox = pred.bboxes
        xy_min = (bbox[0], bbox[1])
        xy_max = (bbox[2], bbox[3])
        image = cv2.rectangle(image, xy_min, xy_max, color, thickness)
        image = cv2.putText(
            img=image,
            text=f"{pred.label_name} {pred.score:.4f}",
            org=(xy_min[0], xy_min[1] + SPACING_PIXELS),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=color,
            thickness=thickness,
        )
    return Image.fromarray(image)


def overlay_colored_masks(
    predictions: List[SegmentationPrediction], image: np.ndarray
) -> Image.Image:
    image = Image.fromarray(image).convert(mode="L")
    masks = [pred.decoded_boolean_mask.astype(np.bool_) for pred in predictions]
    return overlay_masks(image, masks, mask_alpha=0.5, return_pil_image=True)


_OVERLAY_FUNC_MAP = {
    ObjectDetectionPrediction: overlay_bboxes,
    SegmentationPrediction: overlay_colored_masks,
}
