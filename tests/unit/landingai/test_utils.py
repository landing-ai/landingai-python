import os
from pathlib import Path

import pytest

from landingai.common import APIKey
from landingai.exceptions import InvalidApiKeyError
from landingai.utils import load_api_credential


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
