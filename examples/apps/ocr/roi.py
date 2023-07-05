import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import PIL.Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


def draw_region_of_interests(image: PIL.Image.Image) -> Dict[str, Any]:
    # Draw rectangle on the image using streamlit-drawable-canvas
    label_color = (
        st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
    )  # for alpha from 00 to FF
    st.sidebar.write(
        "To draw 4 point polygon left click and draw 3 edges (must in **clockwise order**) and then right click for the 4th point to close the polygon"
    )
    mode = (
        "polygon"
        if st.sidebar.checkbox("4 point polygon (ONLY in Desktop Browser)", False)
        else "rect"
    )
    img_width, img_height = image.size
    st.write(
        "Draw as many boxes as needed in the image below to read the text inside those boxes"
    )

    # Calculate auto scale factor
    canvas_width = min(700, img_width)
    scale_factor = min(1, canvas_width / img_width)
    canvas_height = int(img_height * scale_factor)

    canvas_result = st_canvas(
        fill_color=label_color,  # Orange rectangle with opacity
        stroke_width=2,
        stroke_color="green",  # Green border
        background_image=image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=mode,
        key="canvas_roi",
    )
    if canvas_result.image_data is None:
        return {}
    result = canvas_result.json_data
    result["scale_factor"] = scale_factor
    result["mode"] = mode
    return result


def process_and_display_roi(
    image: PIL.Image.Image, region_of_intrests: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Crop the region of interest from the image, rotate the image if it's vertical, and display the cropped image"""
    assert region_of_intrests, "Canvas result is empty"
    annotations: List[Dict[str, Any]] = region_of_intrests["objects"]
    if len(annotations) == 0:
        return None, None
    scale_factor = region_of_intrests["scale_factor"]
    # Crop the image based on the selected rectangle
    cropped_images = []
    boxes = []
    image_np = np.asarray(image)
    for row in annotations:
        if row["type"] == "rect":
            top = int(row["top"] / scale_factor)
            left = int(row["left"] / scale_factor)
            height = int(row["height"] / scale_factor)
            width = int(row["width"] / scale_factor)
            bottom = int(top + (height * row["scaleY"]))
            right = int(left + (width * row["scaleX"]))
            quad = [[left, top], [right, top], [right, bottom], [left, bottom]]
            cropped_image = image_np[top:bottom, left:right]
        elif row["type"] == "path":
            points = row["path"]
            if len(points) < 4:
                st.error(
                    "Number of vertices in polygon needs to be 4, Delete the polygon and redraw"
                )
                return None, None
            quad = np.array([point[1:] for point in points[:4]])
            quad = (quad // scale_factor).tolist()
            cropped_image = get_minarea_rect_crop(image_np, quad)
        else:
            logging.debug(f"Ignore unknown annotation type {row['type']} for {row}")
            continue

        boxes.append(quad)
        # Rotate image if its vertical
        if cropped_image.shape[0] > cropped_image.shape[1]:
            cropped_image = np.rot90(cropped_image, -1)
        cropped_images.append(cropped_image)

    st.image(
        cropped_images,
        channels="RGB",
        caption=[f"ROI Image{n+1}" for n in range(len(cropped_images))],
    )
    return cropped_images, boxes


# TODO: consider move it to landingai.transform
def get_minarea_rect_crop(
    image: np.ndarray, points: List[List[Tuple[int, int]]]
) -> np.ndarray:
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(image, np.array(box))
    return crop_img


# TODO: consider move it to landingai.transform
def get_rotate_crop_image(
    image: np.ndarray, points: List[List[Tuple[int, int]]]
) -> np.ndarray:
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        image,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img
