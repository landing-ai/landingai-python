import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from landingai.common import (
    APICredential,
    APIKey,
    ObjectDetectionPrediction,
    SegmentationPrediction,
    decode_bitmap_rle,
)
from landingai.exceptions import InvalidApiKeyError


def test_load_api_key_from_constructor():
    key = APIKey(api_key="land_sk_1234")
    assert key.api_key == "land_sk_1234"
    with pytest.raises(InvalidApiKeyError):
        APIKey(api_key="1234")


def test_load_api_key_from_env_var():
    os.environ["landingai_api_key"] = "1234"
    with pytest.raises(InvalidApiKeyError):
        APIKey()
    os.environ["landingai_api_key"] = "land_sk_1234"
    key = APIKey()
    assert key.api_key == "land_sk_1234"
    del os.environ["landingai_api_key"]


def test_load_api_key_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        """
                        LANDINGAI_API_KEY="land_sk_2222"
                        LANDINGAI_API_SECRET="abcd"
                        """
    )
    # Overwrite the default env_prefix to avoid conflict with the real .env
    APIKey.__config__.env_file = str(env_file)
    for field in APIKey.__fields__.values():
        APIKey.__config__.prepare_field(field)
    # Start testing
    credential = APIKey(_env_file=str(env_file))
    assert credential.api_key == "land_sk_2222"
    env_file.unlink()
    with pytest.raises(ValidationError):
        APIKey()
    # reset back to the default config
    APIKey.__config__.env_file = ".env"
    env_file.unlink()


def test_load_credential():
    os.environ["landingai_api_key"] = "1234"
    os.environ["landingai_api_secret"] = "abcd"
    credential = APICredential()
    assert credential.api_key == "1234"
    assert credential.api_secret == "abcd"
    del os.environ["landingai_api_key"]
    del os.environ["landingai_api_secret"]


def test_load_credential_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        """
                        TEST_LANDINGAI_API_KEY="1234"
                        TEST_LANDINGAI_API_SECRET="abcd"
                        """
    )
    # Overwrite the default env_prefix to avoid conflict with the real .env
    APICredential.__config__.env_prefix = "TEST_LANDINGAI_"
    APICredential.__config__.env_file = str(env_file)
    for field in APICredential.__fields__.values():
        APICredential.__config__.prepare_field(field)
    # Start testing
    credential = APICredential(_env_file=str(env_file))
    assert credential.api_key == "1234"
    assert credential.api_secret == "abcd"
    env_file.unlink()
    with pytest.raises(ValidationError):
        APICredential()


def test_decode_bitmap_rle():
    encoded_mask = "2N3Z2N5Z"
    encoding_map = {"Z": 0, "N": 1}
    decoded_mask = decode_bitmap_rle(encoded_mask, encoding_map)
    assert decoded_mask == [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_segmentation_prediction_properties():
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
    assert prediction.num_predicted_pixels == 4


def test_object_detection_prediction_properties():
    label_index = 3
    prediction = ObjectDetectionPrediction(
        id="123",
        label_index=label_index,
        label_name="class1",
        score=0.5,
        bboxes=(1, 2, 31, 42),
    )
    assert prediction.num_predicted_pixels == 1200
