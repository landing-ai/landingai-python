from typing import Any, Dict

import numpy as np
import pytest

from landingai.common import SegmentationPrediction


@pytest.fixture
def seg_mask_validator():
    def assert_seg_mask(pred: SegmentationPrediction, expected: Dict[str, Any]):
        assert pred.label_name == expected["label_name"]
        assert pred.label_index == expected["label_index"]
        np.testing.assert_almost_equal(
            pred.score, expected["score"], decimal=3, err_msg="SEG score mismatch"
        )
        assert pred.num_predicted_pixels == expected["num_predicted_pixels"]
        assert (
            pred.percentage_predicted_pixels == expected["percentage_predicted_pixels"]
        )
        assert pred.decoded_boolean_mask.shape == expected["mask_shape"]
        assert np.unique(pred.decoded_boolean_mask).tolist() == [0, 1]
        assert np.unique(pred.decoded_index_mask).tolist() == [0, pred.label_index]
        if "encoded_mask" in expected:
            assert pred.encoded_mask == expected["encoded_mask"]

    return assert_seg_mask
