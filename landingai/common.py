import re
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, BaseSettings


class Credential(BaseSettings):
    # TODO: support env vars
    api_key: str
    api_secret: str


class Prediction(BaseModel):
    # TODO: add docs
    id: str
    score: float
    label_name: str
    label_index: int


class ObjectDetectionPrediction(Prediction):
    # TODO: add docs
    # xmin, ymin, xmax, ymax
    bboxes: Tuple[int, int, int, int]


class SegmentationPrediction(Prediction):
    # TODO: add docs
    encoded_mask: str
    encoding_map: Dict[str, int]
    mask_shape: Tuple[int, int]

    @cached_property
    def decoded_boolean_mask(self) -> np.ndarray:
        # NOTE: 1 and 0, not True and False
        # TODO: add docs
        flattened_bitmap = decode_bitmap_rle(self.encoded_mask, self.encoding_map)
        seg_mask_channel = np.array(flattened_bitmap, dtype=np.uint8).reshape(self.mask_shape)
        return np.zeros(self.mask_shape, dtype=np.uint8) + seg_mask_channel

    @cached_property
    def decoded_index_mask(self) -> np.ndarray:
        # TODO: add docs
        return self.decoded_boolean_mask * self.label_index

    class Config:
        keep_untouched=(cached_property, )


def decode_bitmap_rle(bitmap: str, encoding_map: Dict[str, int]) -> List[int]:
    """
    Decode bitmap string to numpy array
    -----
    bitmap: str
        Single run-length encoded bitmap string. e.g. "5Z3N2Z"
    encoding_map: Dict[str, int]
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
