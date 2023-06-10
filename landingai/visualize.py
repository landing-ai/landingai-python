"""The landingai.visualize module contains functions to visualize the prediction results."""

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landingai.common import (
    ClassificationPrediction,
    ObjectDetectionPrediction,
    OcrPrediction,
    Prediction,
    SegmentationPrediction,
)

_LOGGER = logging.getLogger(__name__)


def overlay_predictions(
    predictions: List[Prediction],
    image: Union[np.ndarray, Image.Image],
    options: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Overlay the prediction results on the input image and return the image with the overlay."""
    if len(predictions) == 0:
        _LOGGER.warning("No predictions to overlay, returning original image")
        return Image.fromarray(image)
    types = {type(pred) for pred in predictions}
    assert len(types) == 1, f"Expecting only one type of prediction, got {types}"
    pred_type = types.pop()
    overlay_func: Callable[
        [List[Prediction], Union[np.ndarray, Image.Image], Optional[Dict]], Image.Image
    ] = _OVERLAY_FUNC_MAP[pred_type]
    return overlay_func(predictions, image, options)


def overlay_quadrilateral(
    predictions: List[OcrPrediction],
    image: Union[np.ndarray, Image.Image],
    options: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Draw a quadrilateral on the input image and overlay the text on top of the quadrilateral.

    Parameters
    ----------
    predictions
        A list of OcrPrediction, each of which contains the polygon and the predicted text and score.
    image
        The source image to draw the polygon on.
    options
        Options to customize the drawing. Currently, no options are supported.

    Returns
    -------
    Image
        The image with the polygon and text drawn.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    texts = [pred.text for pred in predictions]
    boxes = [pred.text_location for pred in predictions]
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    if texts is None or len(texts) != len(boxes):
        texts = [None] * len(boxes)
    for _, (box, txt) in enumerate(zip(boxes, texts)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        img_right_text = _draw_box_text((w, h), box, txt)
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return img_show


def overlay_bboxes(
    predictions: List[ObjectDetectionPrediction],
    image: Union[np.ndarray, Image.Image],
    options: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Draw bounding boxes on the input image and return the image with bounding boxes drawn.
    The bounding boxes are drawn using the bbox-visualizer package.

    Parameters
    ----------
    predictions
        A list of ObjectDetectionPrediction, each of which contains the bounding box and the predicted class.
    image
        The source image to draw the bounding boxes on.
    options
        Options to customize the drawing. Currently, it supports the following options:
        1. bbox_style: str, the style of the bounding box.
            - "default": draw a rectangle with the label right on top of the rectangle. (default option)
            - "flag": draw a vertical line connects the detected object and the label. No rectangle is drawn.
            - "t-label": draw a rectangle with a vertical line on top of the rectangle, which points to the label.
            For more information, see https://github.com/shoumikchow/bbox-visualizer
        2. draw_label: bool, default True. If False, the label won't be drawn. This option is only valid when bbox_style is "default". This option is ignored otherwise.

    Returns
    -------
    Image.Image
        The image with bounding boxes drawn.

    Raises
    ------
    ValueError
        When the value of bbox_style is not supported.
    """
    import bbox_visualizer as bbv

    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if options is None:
        options = {}
    bbox_style = options.get("bbox_style", "default")
    for pred in predictions:
        bbox = pred.bboxes
        label = f"{pred.label_name} | {pred.score:.4f}"
        if bbox_style == "flag":
            image = bbv.draw_flag_with_label(image, label, bbox)
        else:
            draw_bg = options.get("draw_bg", True)
            label_at_top = options.get("top", True)
            image = bbv.draw_rectangle(image, pred.bboxes)
            if bbox_style == "default" and not options.get("no_label", False):
                image = bbv.add_label(
                    image, label, bbox, draw_bg=draw_bg, top=label_at_top
                )
            elif bbox_style == "t-label":
                image = bbv.add_T_label(image, label, bbox, draw_bg=draw_bg)
            else:
                raise ValueError(
                    f"Unknown bbox_style: {bbox_style}. Supported types are: default (rectangle), flag, t-label. Fore more information, see https://github.com/shoumikchow/bbox-visualizer."
                )
    return Image.fromarray(image)


def overlay_colored_masks(
    predictions: List[SegmentationPrediction],
    image: Union[np.ndarray, Image.Image],
    options: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Draw colored masks on the input image and return the image with colored masks drawn.

    NOTE:
    - The image is converted to grayscale first, and then the colored masks are drawn on top of it.
    - The colored masks are drawn using the segmentation-mask-overlay package.

    Parameters
    ----------
    predictions
        A list of SegmentationPrediction, each of which contains the segmentation mask and the predicted class.
    image
        The source image to draw the colored masks on.
    options
        Options to customize the drawing. Currently, no options are supported.

    Returns
    -------
    Image.Image
        The image with segmented masks drawn.
    """
    from segmentation_mask_overlay import overlay_masks

    if options is None:
        options = {}
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert(mode="L")
    masks = [pred.decoded_boolean_mask.astype(np.bool_) for pred in predictions]
    return cast(
        Image.Image,
        overlay_masks(image, masks, mask_alpha=0.5, return_pil_image=True),
    )


def overlay_predicted_class(
    predictions: List[ClassificationPrediction],
    image: Union[np.ndarray, Image.Image],
    options: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Draw the predicted class on the input image and return the image with the predicted class drawn.

    Parameters
    ----------
    predictions
        A list of ClassificationPrediction, each of which contains the predicted class and the score.
    image
        The source image to draw the colored masks on.
    options
        Options to customize the drawing. Currently, it supports the following options:
        1. text_position: tuple[int, int]. The position of the text relative to the left bottom of the image. The default value is (10, 25).

    Returns
    -------
    Image.Image
        the image with segmented masks drawn.
    """
    if options is None:
        options = {}
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    assert len(predictions) == 1
    text_position = options.get("text_position", (10, 25))
    prediction = predictions[0]
    text = f"{prediction.label_name} {prediction.score:.4f}"
    draw = ImageDraw.Draw(image)
    font = _get_pil_font()
    xy = (text_position[0], image.size[1] - text_position[1])
    box = draw.textbbox(xy=xy, text=text, font=font)
    box = (box[0] - 10, box[1] - 5, box[2] + 10, box[3] + 5)
    draw.rounded_rectangle(box, radius=15, fill="#333333")
    draw.text(xy=xy, text=text, fill="white", font=font)
    return image


def _get_pil_font(font_size: int = 18) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    font = font_manager.FontProperties(family="sans-serif", weight="bold")  # type: ignore
    file = font_manager.findfont(font)  # type: ignore
    assert file, f"Cannot find font file for {font} at {file}"
    return ImageFont.truetype(file, font_size)


_OVERLAY_FUNC_MAP: Dict[
    Type[Prediction],
    Callable[[List[Any], Union[np.ndarray, Image.Image], Optional[Dict]], Image.Image],
] = {
    ObjectDetectionPrediction: overlay_bboxes,
    SegmentationPrediction: overlay_colored_masks,
    ClassificationPrediction: overlay_predicted_class,
    OcrPrediction: overlay_quadrilateral,
}


def _draw_box_text(
    img_size: Tuple[int, int], box: List[Tuple[int, int]], text: Optional[str] = None
) -> np.ndarray:
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if text:
            font = _create_font(text, (box_height, box_width))
            draw_text.text([0, 0], text, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if text:
            font = _create_font(text, (box_width, box_height))
            draw_text.text([0, 0], text, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text


def _create_font(
    txt: str,
    size: Tuple[int, int],
    font_path: str = "./landingai/fonts/default_font_ch_en.ttf",
):
    font_size = int(size[1] * 0.99)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    length = font.getsize(txt)[0]
    if length > size[0]:
        font_size = int(font_size * size[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font
