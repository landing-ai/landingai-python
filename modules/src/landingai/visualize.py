import logging
import re
from typing import Any, Dict, List

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

SPACING_PIXELS = -5


def draw_bboxes(detections: List[Dict[str, Any]], image: np.ndarray) -> np.ndarray:
    "Draw bounding boxes on the input image and return the image with bounding boxes drawn."
    color = (255, 0, 0)
    thickness = 2
    for det in detections:
        logging.debug(f"b: {det}")
        bbox = det["bboxes"]
        xy_min = bbox["xmin"], bbox["ymin"]
        xy_max = bbox["xmax"], bbox["ymax"]
        image = cv2.rectangle(image, xy_min, xy_max, color, thickness)
        image = cv2.putText(
            img=image,
            text=f"{det['label']} {det['confidence_score']:.4f}",
            org=(xy_min[0], xy_min[1] + SPACING_PIXELS),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=color,
            thickness=thickness,
        )
    return image


def decode_bitmap_rle(bitmap: str, encoding_map: Dict[str, int]) -> np.ndarray:
    """
    Decode bitmap string to numpy array
    -----
    bitmap: str
        Single bitmap
    encoding_map: Dict[str, int]
        Dictionary with the enconding used to generate the bitmap. e.g. {'Z':0, 'N':1}

    Return
    -----
    flat_mask: np.ndarray
        Flatten segmentation mask for a single defect
    """
    flat_mask = np.array([])
    bitmap_list = re.split("(Z|N)", bitmap)
    for num, map_letter in zip(*[iter(bitmap_list)] * 2):
        map_number = encoding_map[map_letter]
        flat_mask = np.append(flat_mask, [int(map_number)] * int(num))
    return flat_mask


def generate_segmentation_mask(mask_size, output):
    # seg_mask = np.zeros(image.shape[0:2])
    seg_mask = np.zeros(mask_size)
    encode_map = output.raw.encoding.options.map_
    for _, bitmap in output.raw.bitmaps.items():
        flat_bitmap = decode_bitmap_rle(bitmap.bitmap, encode_map)
        flat_bitmap_index = flat_bitmap * bitmap.label_index
        seg_mask_channel = flat_bitmap_index.reshape(mask_size)
        seg_mask = seg_mask + seg_mask_channel
    return seg_mask


def seg_map_to_rgba(
    seg_map: NDArray, color: tuple[int, int, int] = (255, 0, 0), threshold: float = 0.5
) -> NDArray:
    seg_map[seg_map < threshold] = 0
    seg_map[seg_map >= threshold] = 1
    image = np.zeros((*seg_map.shape, 4), dtype=np.uint8)
    image[:, :, 0] = seg_map * color[0]
    image[:, :, 1] = seg_map * color[1]
    image[:, :, 2] = seg_map * color[2]
    image[:, :, 3] = seg_map * 255
    return image


def seg_map_to_rbga_all_classes(
    seg_map: NDArray, colors: tuple[tuple[int, int, int], ...], threshold: float = 0.5
) -> NDArray:
    seg_map[seg_map < threshold] = 0
    seg_map[seg_map >= threshold] = 1
    image = np.zeros((*seg_map.shape, 4), dtype=np.uint8)
    image[:, :, 0] = colors[0][0]
    image[:, :, 1] = colors[0][1]
    image[:, :, 2] = colors[0][2]
    image[:, :, 3] = 255
    idxs = np.where(seg_map == 1)
    image[idxs[0], idxs[1], 0] = colors[1][0]
    image[idxs[0], idxs[1], 1] = colors[1][1]
    image[idxs[0], idxs[1], 2] = colors[1][2]
    return image


def visualize_segmentation_masks(
    image: Image.Image, masks_image: NDArray
) -> Image.Image:
    pil_image = Image.fromarray(image.astype("uint8"), "RGB")
    pil_image = Image.alpha_composite(pil_image, masks_image)
    return pil_image
