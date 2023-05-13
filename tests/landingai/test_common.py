import os
from pathlib import Path
import numpy as np
from pydantic import ValidationError
import pytest

from landingai.common import APICredential, SegmentationPrediction, decode_bitmap_rle


def test_load_credential():
    with pytest.raises(ValidationError):
        APICredential()
    os.environ["landingai_api_key"] = "1234"
    os.environ["landingai_api_secret"] = "abcd"
    credential = APICredential()
    assert credential.api_key == "1234"
    assert credential.api_secret  == "abcd"


def test_load_credential_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text("""
                        LANDINGAI_API_KEY="1234"
                        LANDINGAI_API_SECRET="abcd"
                        """)
    credential = APICredential(_env_file=str(env_file))
    assert credential.api_key == "1234"
    assert credential.api_secret  == "abcd"


def test_decode_bitmap_rle():
    encoded_mask = "2N3Z2N5Z"
    encoding_map = {"Z": 0, "N": 1}
    decoded_mask = decode_bitmap_rle(encoded_mask, encoding_map)
    assert decoded_mask == [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_segmentation_prediction_get():
    encoded_mask = "2N3Z2N5Z"
    encoding_map = {"Z": 0, "N": 1}
    label_index = 3
    prediction = SegmentationPrediction(
        id="123",
        label_index=label_index,
        label_name="class1",
        score=0.5,
        encoded_mask=encoded_mask,
        encoding_map=encoding_map,
        mask_shape=(3, 4),
    )
    expected = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]).reshape((3, 4))
    np.testing.assert_array_almost_equal(prediction.decoded_boolean_mask, expected)
    np.testing.assert_array_almost_equal(
        prediction.decoded_index_mask, expected * label_index
    )
