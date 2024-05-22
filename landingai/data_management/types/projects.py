from enum import Enum


class ProjectType(str, Enum):
    classification = "classification"
    object_detection = "object-detection"
    segmentation = "segmentation"
