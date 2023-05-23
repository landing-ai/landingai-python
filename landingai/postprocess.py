from collections import defaultdict

import numpy as np

from landingai.common import (
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)


def class_map(predictions: list[Prediction]) -> dict[int, str]:
    """Return a map from the predicted class/label index to the class/label name."""
    return {pred.label_index: pred.label_name for pred in predictions}


def class_counts(predictions: list[Prediction]) -> dict[int, (int, str)]:
    """Compute the distribution of the occurrences of each class.

    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
    (the number of occurrence, class/label name) as the value.
    Example:
        {
            1: (10, "cat"),
            2: (31, "dog"),
        }
    """
    counts = defaultdict(lambda: [0, None])
    for pred in predictions:
        counts[pred.label_index][0] += 1
        counts[pred.label_index][1] = pred.label_name
    return {k: tuple(v) for k, v in counts.items()}


def class_pixel_coverage(
    predictions: list[Prediction],
    coverage_type: str = "relative",
) -> dict[int, (float, str)]:
    """Compute the pixel coverage of each class.

    Supported prediction types are:
    1. SegmentationPrediction
    2. ObjectDetectionPrediction

    It supports two ways to compute the coverage:
    1. "absolute"
    2. "relative"
    See the documentation of the coverage_type for more details.

    Parameters
    ----------
    predictions: a list of predictions. It could come from one or multiple images.
    coverage_type: "absolute" or "relative".
            1. absolute: the number of pixels of each predicted class.
            2. relative: the percentage of pixels that are predicted as the class
               over the sum total number of pixels of every mask.
               NOTE: only "SegmentationPrediction" supports "relative" type.

    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
    (the coverage, class/label name) as the value.

    Example (coverage_type="absolute"):
        {
            0: (23512, "blue"),
            1: (1230, "green"),
            2: (0, "pink"),
        }
    """
    # TODO: add more input validation
    assert (
        type(predictions[0]) == SegmentationPrediction
    ), "Only support SegmentationPrediction for now."
    return segmentation_class_pixel_coverage(predictions, coverage_type)


def od_class_pixel_coverage(
    predictions: list[ObjectDetectionPrediction],
    coverage_type: str = "relative",
):
    raise NotImplementedError()


def segmentation_class_pixel_coverage(
    predictions: list[SegmentationPrediction],
    coverage_type: str = "relative",
) -> dict[int, (float, str)]:
    """Compute the pixel coverage of each class.
    The coverage is defined as the percentage of pixels that are predicted as the class
    over the sum total number of pixels of every mask.

    Parameters
    ----------
    predictions: a list of segmentation predictions. It could come from one or multiple images.

    Returns
    -------
    A map with the predicted class/label index as the key, and a tuple of
    (the coverage percentage, class/label name) as the value.
    NOTE: the sum of the coverage percentage over all classes is not guaranteed
    to be 1.

    Example (coverage_type="relative"):
        {
            0: (0.15, "blue"),
            1: (0.31, "green"),
            2: (0.07, "pink"),
        }
    """
    total_pixels = sum([np.prod(pred.mask_shape) for pred in predictions])
    pixel_counts = defaultdict(lambda: [0, ""])
    for pred in predictions:
        pixel_counts[pred.label_index][0] += pred.num_predicted_pixels
        pixel_counts[pred.label_index][1] = pred.label_name
    coverages = {}
    if coverage_type == "relative":
        total_pixels = sum([np.prod(pred.mask_shape) for pred in predictions])
    else:
        total_pixels = 1
    for label_index, (num_pixels, class_name) in pixel_counts.items():
        coverages[label_index] = (num_pixels / total_pixels, class_name)
    return coverages
