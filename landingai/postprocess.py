from collections import defaultdict


import numpy as np

from landingai.common import SegmentationPrediction


def segmentation_class_pixel_coverage(
    predictions: list[SegmentationPrediction],
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

    Example:
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
    for label_index, (num_pixels, class_name) in pixel_counts.items():
        coverages[label_index] = (num_pixels / total_pixels, class_name)
    return coverages
