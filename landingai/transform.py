"""Module for image transformations."""

from typing import List, Tuple

import cv2
import numpy as np
import PIL.Image


def crop_rotated_rectangle(
    image: PIL.Image.Image,
    rect: List[Tuple[int, int]],
    angle: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Crop the input image based on the rotated rectangle.
    The rectangle is calculated based on the rotated rectangle's corners.

    Parameters
    ----------
    img
       the input image to be cropped
    rect
        the unrotated rectangle (in parallel with the edges of the image)
    angle
        the angle of the rotated rectangle
    Returns
    -------
    Tuple[np.ndarray, List[Tuple[int, int]]]
        the cropped image and the coordinates of the rotated rectangle
    """
    # rot_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    # corners = cv2.transform(np.array(rect)[None], rot_matrix)
    # quad_box = corners[0].tolist()
    # return get_minarea_rect_crop(img, corners), quad_box
    [[left, top], [right, top], [right, bottom], [left, bottom]] = rect
    center = ((left + right) / 2, (top + bottom) / 2)
    img_np = np.asarray(image)
    shape = (img_np.shape[1], img_np.shape[0])

    matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    img_np = cv2.warpAffine(src=img_np, M=matrix, dsize=shape)
    width, height = rect[1][0] - rect[0][0], rect[3][1] - rect[0][1]
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)

    img_np = img_np[y : y + height, x : x + width]
    corners = cv2.transform(np.array(rect)[None], matrix)
    quad_box: List[Tuple[int, int]] = corners[0].tolist()

    return img_np, quad_box
