import math
from collections import defaultdict
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
    total_pixels: int = sum([math.prod(pred.mask_shape) for pred in predictions])
    pixel_counts: Dict[int, List] = defaultdict(lambda: [0, ""])
    for pred in predictions:
        pixel_counts[pred.label_index][0] += pred.num_predicted_pixels
        pixel_counts[pred.label_index][1] = pred.label_name
    coverages: Dict[int, Tuple[float, str]] = {}
    if coverage_type == "relative":
        total_pixels = sum([math.prod(pred.mask_shape) for pred in predictions])
    else:
        total_pixels = 1
    for label_index, (num_pixels, class_name) in pixel_counts.items():
        coverages[label_index] = (num_pixels / total_pixels, class_name)
    return coverages
