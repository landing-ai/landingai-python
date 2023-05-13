import re
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, BaseSettings


class Credential(BaseSettings):
    """Landing AI API credential of a particular LandingLens user."""

    api_key: str
    api_secret: str


class Prediction(BaseModel):
    """The base/parent prediction class that stores the common shared properties of a prediction."""

    """A unique string identifier (UUID) for this prediction."""
    id: str

    """The confidence score of this prediction."""
    score: float

    """The predicted label name."""
    label_name: str

    """The predicted label index.
    A label index is an unique integer identifies a label in your label book.
    See https://support.landing.ai/docs/manage-label-book for more details.
    """
    label_index: int


class ObjectDetectionPrediction(Prediction):
    """A single bounding box prediction for an image.
    It includes a predicted bounding box (xmin, ymin, xmax, ymax), confidence score and the predicted label.
    """

    """A tuple of (xmin, ymin, xmax, ymax) of the predicted bounding box."""
    bboxes: Tuple[int, int, int, int]


class SegmentationPrediction(Prediction):
    """A single segmentation mask prediction for an image.
    It includes a predicted segmentation mask, confidence score and the predicted label.
    """

    """A run-length encoded bitmap string."""
    encoded_mask: str

    """A map that is used to generate the encoded_mask. e.g. {'Z':0, 'N':1}
    The key is the character in the encoded_mask, the value is the bit value.
    """
    encoding_map: Dict[str, int]

    """The shape of the decoded 2-dimensional segmentation mask. e.g. (1024, 1024)"""
    mask_shape: Tuple[int, int]

    @cached_property
    def decoded_boolean_mask(self) -> np.ndarray:
        """Decoded boolean segmentation mask.
        It is a 2-dimensional numpy array with 0s and 1s.
        1 means the pixel is the predicted class, 0 means the pixel is not.
        """
        flattened_bitmap = decode_bitmap_rle(self.encoded_mask, self.encoding_map)
        seg_mask_channel = np.array(flattened_bitmap, dtype=np.uint8).reshape(
            self.mask_shape
        )
        return seg_mask_channel

    @cached_property
    def decoded_index_mask(self) -> np.ndarray:
        """Decoded index segmentation mask.
        It is a 2-dimensional numpy array with 0s and the number of the predicted class index.
        This is useful if you want to overlay multiple segmentation masks into one.
        """
        return self.decoded_boolean_mask * self.label_index

    class Config:
        keep_untouched = (cached_property,)


def decode_bitmap_rle(bitmap: str, encoding_map: Dict[str, int]) -> List[int]:
    """
    Decode bitmap string to numpy array

    Parameters
    ----------
    bitmap:
        Single run-length encoded bitmap string. e.g. "5Z3N2Z"
    encoding_map:
        Dictionary with the enconding used to generate the bitmap. e.g. {'Z':0, 'N':1}

    Return
    -----
    A flattened segmentation mask (with 0s and 1s) for a single class.
    """
    flat_mask = []
    bitmap_list = re.split("(Z|N)", bitmap)
    for num, map_letter in zip(*[iter(bitmap_list)] * 2):
        map_number = encoding_map[map_letter]
        flat_mask.extend([int(map_number)] * int(num))
    return flat_mask
