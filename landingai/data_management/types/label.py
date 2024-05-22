from enum import Enum


class LabelType(str, Enum):
    classification = "classification"
    object_detection = "bounding_box"
    segmentation = "segmentation"
