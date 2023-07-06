"""Module for common utility functions."""
import io
from typing import Optional, Union

import numpy as np
import PIL.Image
from pydantic import ValidationError

from landingai.common import APICredential, APIKey
from landingai.exceptions import InvalidApiKeyError


def serialize_image(image: Union[np.ndarray, PIL.Image.Image]) -> bytes:
    """Serialize the input image into bytes."""
    if image is None or (isinstance(image, np.ndarray) and len(image) == 0):
        raise ValueError(f"Input image must be non-emtpy, but got: {image}")
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)

    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    buffer_bytes = img_buffer.getvalue()
    img_buffer.close()
    return buffer_bytes


def load_api_credential(api_key: Optional[str] = None) -> Union[APIKey, APICredential]:
    """Load API credential from different sources.
    The API credential can be provided as arguments or loaded from the environment variables or .env file.
    The priority is: arguments > environment variables > .env file.
    The output is an APIKey (v2 key) or APICredential (v1 key) object.
    """
    if api_key is not None:
        return APIKey(api_key=api_key)
    else:
        # Load from environment variables or .env file
        try:
            return APIKey()
        except (InvalidApiKeyError, ValidationError) as e:
            try:
                return APICredential()
            except Exception:
                raise InvalidApiKeyError(
                    "API key is not provided or it's invalid."
                ) from e
