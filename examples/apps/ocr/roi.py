from typing import Any, Dict, List, Tuple


import numpy as np
import pandas as pd
import PIL.Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from landingai.transform import crop_rotated_rectangle


def draw_region_of_interests(image: PIL.Image.Image) -> Dict[str, Any]:
    # Draw rectangle on the image using streamlit-drawable-canvas
    label_color = (
        st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
    )  # for alpha from 00 to FF
    mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"
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
    return result


def process_and_display_roi(
    image: PIL.Image.Image,
    region_of_intrests: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Crop the region of interest from the image, rotate the image if it's vertical, and display the cropped image"""
    assert region_of_intrests, "Canvas result is empty"
    annotation = pd.json_normalize(region_of_intrests["objects"])
    if len(annotation) == 0:
        return None, None
    roi_df = annotation[["top", "left", "width", "height", "scaleX", "scaleY", "angle"]]
    st.table(roi_df)
    scale_factor = region_of_intrests["scale_factor"]
    # Crop the image based on the selected rectangle
    cropped_images = []
    boxes = []
    for _, row in roi_df.iterrows():
        top = int(row["top"] / scale_factor)
        left = int(row["left"] / scale_factor)
        height = int(row["height"] / scale_factor)
        width = int(row["width"] / scale_factor)
        bottom = int(top + (height * row["scaleY"]))
        right = int(left + (width * row["scaleX"]))
        angle = row["angle"]
        rect = [[left, top], [right, top], [right, bottom], [left, bottom]]
        cropped_image, quad_box = crop_rotated_rectangle(image, rect, angle)
        boxes.append(quad_box)
        # cropped_image = image[top:bottom, left:right]
        # boxes.append([[left, top], [left + width, top], [left + width, top + height], [left, top + height]])

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
