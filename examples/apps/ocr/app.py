import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

import extra_streamlit_components as stx
import numpy as np
import streamlit as st
from PIL import Image
from roi import draw_region_of_interests, process_and_display_roi

from landingai.predict import OcrPredictor, serialize_rois
from landingai.st_utils import (
    get_api_key_or_use_default,
    get_default_api_key,
    render_api_config_form,
    render_svg,
    setup_page,
)
from landingai.visualize import overlay_predictions

setup_page(page_title="LandingLens OCR")


class DetModel(str, Enum):
    AUTO_DETECT = "multi-text"
    MANUAL_ROI = "single-text"


# Streamlit app code
def main():
    render_svg(Path("./examples/apps/ocr/static/LandingLens_OCR_logo.svg").read_text())
    st.sidebar.title("Configuration")
    with st.sidebar:
        render_api_config_form()

    chosen_id = stx.tab_bar(
        data=[
            stx.TabBarItemData(id=1, title="Upload an image", description=""),
            stx.TabBarItemData(id=2, title="Take a photo", description=""),
        ],
        default=1,
    )
    if chosen_id == "1":
        image_file = st.file_uploader("File Uploader")
    elif chosen_id == "2":
        image_file = st.camera_input("Camera View")
    else:
        st.error(f"Unknown tab id: {chosen_id}")

    # Add radio button for image selection
    detection_mode = st.radio(
        label="OCR Mode",
        options=[
            DetModel.AUTO_DETECT.name,
            DetModel.MANUAL_ROI.name,
        ],
        help="Auto Detect - Ideal for reading text in multiple lines or single line where each line is a collection of words. "
        "Examples are reading text in a document, print, label or image where location of text is dynamic."
        "Manual ROI - For more complex use cases draw a ROI for every word to be read. The text location has to be the same for every image ",
    )
    st.markdown(
        ":blue[Auto Detect] - Ideal for reading text in multiple lines or single line where each line is a stream of characters."
        "Examples are reading text in a document, print, label or image where location of text is dynamic."
    )
    st.markdown(
        ":blue[Manual ROI] - Ideal for For more complex use cases where user draws a box encapsulating every word to be read.  "
        "Used when auto detect modes fail to locate text reliably. The text location has to be the same for every image "
    )

    # Add image viewer
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        img_width, img_height = image.size
        image_np = np.asarray(image)
        st.write(f"Image size : {img_width} x {img_height}")
        # Check if the user selected the "Cropped Region" option
        if detection_mode == DetModel.MANUAL_ROI.name:
            # Update image object which will be used in inference
            roi_result = draw_region_of_interests(image)
            if roi_result:
                input_images, boxes = process_and_display_roi(image, roi_result)
            else:
                input_images = None
                boxes = None
        else:
            st.image(image, channels="RGB", caption="Uploaded Image")
            input_images = image_np
            boxes = None
        # Slider for choosing a confidence threshold
        threshold = st.slider(
            "Recognition Threshold",
            0.0,
            1.0,
            0.5,
            key="th_slider",
        )
        mode = DetModel[detection_mode].value
        key = get_api_key_or_use_default()
        # Run ocr on the whole image
        if input_images is not None and st.button("Run"):
            predictor = OcrPredictor(
                threshold=float(threshold),
                api_key=key,
            )
            begin = time.perf_counter()
            logging.info(
                f"Running OCR prediction in {mode} mode with threshold {threshold} and rois: {boxes}"
            )
            preds = predictor.predict(image, mode=mode, regions_of_interest=boxes)
            tak_time = time.perf_counter() - begin
            display_image = overlay_predictions(preds, image_np)
            st.image(
                display_image,
                channels="RGB",
                caption=[f"Inference Time = {tak_time:.3f} sec"],
            )
            st.divider()
            st.subheader("Prediction Result (in JSON):")
            json_result = [pred.dict() for pred in preds]
            for pred in json_result:
                pred["location"] = [
                    {"x": loc[0], "y": loc[1]} for loc in pred["location"]
                ]
            st.json(json_result)

        _render_curl_command(mode, boxes, key)


def _render_curl_command(
    mode: str,
    rois: Optional[List[List[int]]] = None,
    api_key: Optional[str] = None,
) -> str:
    st.divider()
    # Build curl command str
    curl_command_str = f"""
    curl --location --request POST '{OcrPredictor._url}' \\
     --header 'Content-Type: multipart/form-data' \\
     --header 'apikey: YOUR_API_KEY' \\"""
    if api_key and api_key != get_default_api_key():
        curl_command_str = curl_command_str.replace("YOUR_API_KEY", api_key)
    if rois:
        curl_command_str += f"\n     --form 'rois={serialize_rois(rois, mode)}' \\"
    curl_command_str += "\n     --form 'images=@\"YOUR_FILE_PATH\"'"
    # Display curl command
    st.subheader("Run Inference with cURL command")
    st.caption(
        """Instructions for composing a valid curl command:
 1. Enter your api key in the sidebar.
 2. Replace 'YOUR_FILE_PATH' with the path to your local image file."""
    )
    st.code(curl_command_str)


# Run the app
if __name__ == "__main__":
    main()
