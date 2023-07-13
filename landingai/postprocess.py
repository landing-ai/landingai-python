import math
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Sequence, Tuple, Union, cast

import numpy as np
import PIL.Image
from PIL.Image import Image

from landingai.common import (
    ClassificationPrediction,
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)


def rescale_bboxes_by_image_size(
    predictions: Sequence[ObjectDetectionPrediction],
    from_image: Image,
    to_image: Image,
) -> Sequence[ObjectDetectionPrediction]:
    """Rescale the bounding boxes of each ObjectDetectionPrediction in a list based on the size of the reference images.
    NOTE: this operation is NOT in-place. The input predictions will remain the same.

    Parameters
    ----------
    predictions: a list of ObjectDetectionPrediction to be rescaled.
    from_image: the image that serves the denominator of the scale factor. The input bboxes is at the same scale of this image.
    to_image: the image that serves the numerator of the scale factor. The ouput bboxes will be at the same scale of this image.

    Returns
    -------
    A list of ObjectDetectionPrediction where the bboxes has been rescaled.
    """
    scale_factor = (
        to_image.size[1] / from_image.size[1],
        to_image.size[0] / from_image.size[0],
    )
    return rescale_bboxes(predictions, scale_factor)


def rescale_bboxes(
    predictions: Sequence[ObjectDetectionPrediction],
    scale_factor: Union[Tuple[float, float], float],
) -> Sequence[ObjectDetectionPrediction]:
    """Rescale the bounding boxes of each ObjectDetectionPrediction in a list based on the scale factor.
    NOTE: this operation is NOT in-place. The input predictions will remain the same.

    Parameters
    ----------
    predictions: a list of ObjectDetectionPrediction to be rescaled.
    scale_factor: the scale factor that will be applied to the predictions. The scale factors are (height, width) if it's a tuple.

    Returns
    -------
    A list of ObjectDetectionPrediction where the bboxes has been rescaled.
    """
    if isinstance(scale_factor, float):
        scale_factor = (scale_factor, scale_factor)
    assert scale_factor[0] > 0 and scale_factor[1] > 0, "Scale factor must be > 0"
    height_scale, width_scale = scale_factor
    return [
        ObjectDetectionPrediction(
            id=pred.id,
            score=pred.score,
            label_index=pred.label_index,
            label_name=pred.label_name,
            bboxes=(
                math.floor(pred.bboxes[0] * width_scale),  # xmin
                math.floor(pred.bboxes[1] * height_scale),  # ymin
                math.ceil(pred.bboxes[2] * width_scale),  # xmax
                math.ceil(pred.bboxes[3] * height_scale),  # ymax
            ),
        )
        for pred in predictions
    ]


def crop(
    predictions: Sequence[Prediction], image: Union[np.ndarray, Image]
) -> Sequence[Image]:
    """Crop the image based on the bounding boxes in the predictions.

    NOTE: Currently, only ObjectDetectionPrediction is supported. If other types of predictions are passed in, a ValueError will be raised.

    Parameters
    ----------
    predictions: a list of ObjectDetectionPrediction, each of which will be used to crop the image.
    image: the input image to be cropped from.

    Returns
    -------
    A list of cropped images.
    """
    if len(predictions) == 0:
        return []
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    output = []
    for pred in predictions:
        if not isinstance(pred, ObjectDetectionPrediction):
            raise ValueError(
                f"Only ObjectDetectionPrediction is supported but {type(pred)} is found."
            )
        output.append(image.crop(pred.bboxes))
    return output


def class_map(predictions: Sequence[ClassificationPrediction]) -> Dict[int, str]:
    """Return a map from the predicted class/label index to the class/label name."""
    return {pred.label_index: pred.label_name for pred in predictions}


def class_counts(
    predictions: Sequence[ClassificationPrediction],
) -> Dict[int, Tuple[int, str]]:
    """Compute the distribution of the occurrences of each class.

    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
        (the number of occurrence, class/label name) as the value.
        ```
        Example:
            {
                1: (10, "cat"),
                2: (31, "dog"),
            }
        ```
    """
    counts: Dict[int, List[Union[int, str]]] = defaultdict(lambda: [0, ""])
    for pred in predictions:
        counts[pred.label_index][0] = cast(int, counts[pred.label_index][0]) + 1
        counts[pred.label_index][1] = pred.label_name
    return {k: (cast(int, v[0]), cast(str, v[1])) for k, v in counts.items()}


def class_pixel_coverage(
    predictions: Sequence[ClassificationPrediction],
    coverage_type: str = "relative",
) -> Dict[int, Tuple[float, str]]:
    """Compute the pixel coverage of each class.

    Supported prediction types are:
    - SegmentationPrediction
    - ObjectDetectionPrediction

    It supports two ways to compute the coverage:
    - "absolute"
    - "relative"
    See the documentation of the coverage_type for more details.

    Parameters
    ----------
    predictions: a list of predictions. It could come from one or multiple images.
    coverage_type: "absolute" or "relative".
            - Absolute: The number of pixels of each predicted class.
            - Relative: The percentage of pixels that are predicted as the class
               over the sum total number of pixels of every mask. The only project type that supports "relative" is "SegmentationPrediction".


    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
        (the coverage, class/label name) as the value.
        ```
        Example (coverage_type="absolute"):
            {
                0: (23512, "blue"),
                1: (1230, "green"),
                2: (0, "pink"),
            }
        ```
    """
    assert isinstance(
        predictions[0], SegmentationPrediction
    ), "Only support SegmentationPrediction for now."
    predictions = cast(List[SegmentationPrediction], predictions)
    return segmentation_class_pixel_coverage(predictions, coverage_type)


def od_class_pixel_coverage(
    predictions: Sequence[ObjectDetectionPrediction],
    coverage_type: str = "relative",
) -> Dict[int, Tuple[float, str]]:
    raise NotImplementedError()


def segmentation_class_pixel_coverage(
    predictions: Sequence[SegmentationPrediction],
    coverage_type: str = "relative",
) -> Dict[int, Tuple[float, str]]:
    """Compute the pixel coverage of each class.
    The coverage is defined as the percentage of pixels that are predicted as the class
    over the sum total number of pixels of every mask.
    If the predictions are from multiple images, the coverage is the average coverage across all images.

    Parameters
    ----------
    predictions: A list of segmentation predictions. It could come from one or multiple images.

    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
        (the coverage percentage, class/label name) as the value.
        Note: The sum of the coverage percentage over all classes is not guaranteed
        to be 1.
        ```
        Example (coverage_type="relative"):
            {
                0: (0.15, "blue"),
                1: (0.31, "green"),
                2: (0.07, "pink"),
            }
        ```
    """
    pixel_coverages = []
    label_map = {}
    for pred in predictions:
        coverage = cast(float, pred.num_predicted_pixels)
        if coverage_type == "relative":
            coverage /= math.prod(pred.mask_shape)
        pixel_coverages.append((pred.label_index, coverage))
        label_map[pred.label_index] = pred.label_name

    sorted(pixel_coverages, key=lambda x: x[0])
    coverage_by_label: Dict[int, Tuple[float, str]] = {}
    for label_index, group in groupby(pixel_coverages, key=lambda x: x[0]):
        cov_vals = [item[1] for item in list(group)]
        avg_coverage = sum(cov_vals) / len(cov_vals) if len(cov_vals) > 0 else 0
        coverage_by_label[label_index] = (avg_coverage, label_map[label_index])
    return coverage_by_label
