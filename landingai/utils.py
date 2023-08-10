"""Module for common utility functions."""
import io
import logging
import os
from typing import Optional, Union

import numpy as np
import PIL.Image
from pydantic import ValidationError

from landingai.common import APIKey
from landingai.exceptions import InvalidApiKeyError
from landingai.timer import Timer

_LLENS_SUPPORTED_IMAGE_FORMATS = ["JPG", "JPEG", "PNG", "BMP"]
_DEFAULT_FORMAT = "JPEG"
_IMG_SERIALIZATION_FORMAT_KEY = "DEFAULT_IMAGE_SERIALIZATION_FORMAT"


_LOGGER = logging.getLogger(__name__)


@Timer(name="serialize_image", log_fn=_LOGGER.debug)
def serialize_image(image: Union[np.ndarray, PIL.Image.Image]) -> bytes:
    """Serialize the input image into bytes of an image file.

    The image file encoding format is set to "JPEG" for optimal latency.
    For RGBA images, the encoding format will be `PNG` to preserve transparency.
    For palettized image, i.e. mode `P` or `PA`, the image will be converted to RGB before encoding.
    In addition, the image file encoding format can be overwritten by the environment variable `DEFAULT_IMAGE_SERIALIZATION_FORMAT`.
    Supported image serialization formats are: JPEG, PNG, BMP.
    """
    if image is None or (isinstance(image, np.ndarray) and len(image) == 0):
        raise ValueError(f"Input image must be non-emtpy, but got: {image}")
    format = _resolve_serialization_format(image)
    _LOGGER.debug("Use %s as the serialization format.", format)
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    if image.mode in ["P", "PA"]:
        image = image.convert("RGB")
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=format)
    buffer_bytes = img_buffer.getvalue()
    img_buffer.close()
    return buffer_bytes


def _resolve_serialization_format(image: Union[np.ndarray, PIL.Image.Image]) -> str:
    default_format = os.getenv(_IMG_SERIALIZATION_FORMAT_KEY, _DEFAULT_FORMAT)
    assert default_format.upper() in _LLENS_SUPPORTED_IMAGE_FORMATS
    if isinstance(image, np.ndarray):
        return default_format

    assert isinstance(image, PIL.Image.Image)
    if image.mode == "RGBA":
        return "PNG"
    return "JPEG"


def load_api_credential(api_key: Optional[str] = None) -> APIKey:
    """Load API credential from different sources.

    Parameters
    ----------
    api_key:
        The API key argument to be passed in, by default None.
        The API key can be provided as arguments or loaded from the environment variables or .env file.
        The api key loading priority is: arguments > environment variables > .env file.

    Returns
    -------
    APIKey
        An APIKey (v2 key) instance.
    """
    if api_key is not None:
        return APIKey(api_key=api_key)
    else:
        # Load from environment variables or .env file
        try:
            return APIKey()
        except ValidationError as e:
            raise InvalidApiKeyError(
                "API key is either not provided or invalid."
            ) from e
