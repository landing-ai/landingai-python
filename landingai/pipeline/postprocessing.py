from typing import Dict, Sequence, cast

from landingai.pipeline.frameset import FrameSet
from landingai.postprocess import class_counts
from landingai.common import ClassificationPrediction


def get_class_counts(
    frs: FrameSet, add_id_to_classname: bool = False
) -> Dict[str, int]:
    """This method returns the number of occurrences of each detected class in the FrameSet.

    Parameters
    ----------
    add_id_to_classname : bool, optional
        By default, detections with the same class names and different defect
        id will be counted as the same. Set to True if you want to count them
        separately

    Returns
    -------
    Dict[str, int]
        A dictionary with the counts
        ```
        Example:
            {
                "cat": 10,
                "dog": 3
            }
        ```
    """
    counts = {}
    for frame in frs.frames:
        # Here is a sample return from class_counts: {1: (3, 'Heart'), 3: (3, 'Club'), 4: (3, 'Spade'), 2: (3, 'Diamond')}
        if frame.predictions._inner_type == "OcrPrediction":
            raise TypeError("Can't count classes for OcrPredictor")
        predictions = cast(Sequence[ClassificationPrediction], frame.predictions)
        for k, v in class_counts(predictions).items():
            if add_id_to_classname:  # This is useful if class names are not unique
                class_name = f"{v[1]}_{k}"
            else:
                class_name = v[1]
            if class_name not in counts:
                counts[class_name] = v[0]
            else:
                counts[class_name] += v[0]
    return counts
