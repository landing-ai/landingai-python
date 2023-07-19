import io
import os
from pathlib import Path

import PIL.Image
import numpy as np
import pytest

from landingai.common import APIKey
from landingai.exceptions import InvalidApiKeyError
from landingai.utils import _DEFAULT_FORMAT, load_api_credential, serialize_image


def test_load_api_credential_invalid_key():
    with pytest.raises(InvalidApiKeyError):
        load_api_credential()
    with pytest.raises(InvalidApiKeyError):
        load_api_credential(api_key="fake_key")
    with pytest.raises(InvalidApiKeyError):
        os.environ["landingai_api_key"] = "1234"
        load_api_credential()


def test_load_api_credential_from_constructor():
    actual = load_api_credential(api_key="land_sk_1234")
    assert actual.api_key == "land_sk_1234"


def test_load_api_credential_from_env_var():
    os.environ["landingai_api_key"] = "land_sk_123"
    actual = load_api_credential()
    assert actual.api_key == "land_sk_123"
    del os.environ["landingai_api_key"]


def test_load_api_credential_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        """
                        LANDINGAI_API_KEY="land_sk_12345"
                        """
    )
    # Overwrite the default env_prefix to avoid conflict with the real .env
    APIKey.__config__.env_file = str(env_file)
    actual = load_api_credential()
    assert actual.api_key == "land_sk_12345"
    # reset back to the default config
    APIKey.__config__.env_file = ".env"
    env_file.unlink()


@pytest.mark.parametrize(
    "expected",
    [
        PIL.Image.open("tests/data/images/wildfire1.jpeg"),
        PIL.Image.open("tests/data/images/ocr_test.png"),
        PIL.Image.open("tests/data/images/cameraman.tiff"),
        PIL.Image.new("L", (15, 20)),
        PIL.Image.new("RGBA", (35, 25)),
        np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8),
    ],
)
def test_serialize_image(expected):
    serialized_img = serialize_image(expected)
    assert len(serialized_img) > 0
    actual = PIL.Image.open(io.BytesIO(serialized_img))
    if isinstance(expected, PIL.Image.Image):
        assert actual.size == expected.size
        assert actual.mode == expected.mode
        if expected.mode == "RGBA":
            expected_format = "PNG"
        else:
            expected_format = _DEFAULT_FORMAT
        assert actual.format == expected_format
    else:
        assert actual.size == expected.shape[:2][::-1]
