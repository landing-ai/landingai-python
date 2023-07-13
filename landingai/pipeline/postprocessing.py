from typing import Dict

from landingai.pipeline.frameset import FrameSet
from landingai.postprocess import class_counts


def get_class_counts(
    frs: FrameSet, add_id_to_classname: bool = False
) -> Dict[str, int]:
    """This method returns the number of occurrences of each detected class in the FrameSet.

    Parameters
    ----------
    add_id_to_classname : bool, optional
        By default, detections with the same class names and different defect id will be counted as the same. Set to True if you want to count them separately

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
    for i in range(len(frs.frames)):
        # Here is a sample return from class_counts: {1: (3, 'Heart'), 3: (3, 'Club'), 4: (3, 'Spade'), 2: (3, 'Diamond')}
        for k, v in class_counts(frs.frames[i].predictions).items():
            if add_id_to_classname:  # This is useful if class names are not unique
                class_name = f"{v[1]}_{k}"
            else:
                class_name = v[1]
            if class_name not in counts:
                counts[class_name] = v[0]
            else:
                counts[class_name] += v[0]
    return counts
