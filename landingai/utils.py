"""Module for common utility functions."""
import io
from typing import Union

import numpy as np
import PIL


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
