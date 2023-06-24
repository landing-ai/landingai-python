import math
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Sequence, Tuple, Union, cast

from landingai.common import (
    ClassificationPrediction,
    ObjectDetectionPrediction,
    SegmentationPrediction,
)


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
