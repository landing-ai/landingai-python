import time

import extra_streamlit_components as stx
import streamlit_pydantic as sp
import numpy as np
import streamlit as st
from PIL import Image

from examples.apps.ocr.predict import DetModel, OcrPredictor
from examples.apps.ocr.roi import draw_region_of_interests, process_and_display_roi
from landingai.common import APICredential
from landingai.visualize import overlay_predictions


# Streamlit app code
def main():
    st.title("OCR Demo")
    st.sidebar.title("Configuration")
    with st.sidebar:
        credential = sp.pydantic_form(key="api_credential", model=APICredential, submit_label="Save", ignore_empty_values=True)
        if credential:
            st.session_state["credential"] = credential
            st.info("Saved API credential")

    chosen_id = stx.tab_bar(
        data=[
            stx.TabBarItemData(id=1, title="Upload an image", description=""),
            stx.TabBarItemData(id=2, title="Take a photo", description=""),
        ],
        default=1,
    )
    if chosen_id == "1":
        image_file = st.file_uploader(
            "File Uploader", type=["jpg", "jpeg", "png", "bmp"]
        )
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
        help=f"Auto Detect - Ideal for reading text in multiple lines or single line where each line is a collection of words. "
        f"Examples are reading text in a document, print, label or image where location of text is dynamic."
        f"Manual ROI - For more complex use cases draw a ROI for every word to be read. The text location has to be the same for every image ",
    )
    st.markdown(
        f":blue[Auto Detect] - Ideal for reading text in multiple lines or single line where each line is a stream of characters."
        f"Examples are reading text in a document, print, label or image where location of text is dynamic."
    )
    st.markdown(
        f":blue[Manual ROI] - Ideal for For more complex use cases where user draws a box encapsulating every word to be read.  "
        f"Used when auto detect modes fail to locate text reliably. The text location has to be the same for every image "
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
        # Run ocr on the whole image
        if input_images is not None and st.button("Run"):
            if "credential" not in st.session_state:
                st.error("Please enter and save your API credential first")
                return
            api_credential = st.session_state["credential"]
            predictor = OcrPredictor(detection_mode, float(threshold))
            begin = time.perf_counter()
            preds = predictor.predict(input_images, roi_boxes=boxes)
            tak_time = time.perf_counter() - begin
            display_image = overlay_predictions(preds, image_np)
            st.image(
                display_image,
                channels="RGB",
                caption=[f"Inference Time = {tak_time:.3f} sec"],
            )
            st.json([pred.json() for pred in preds])


# Run the app
if __name__ == "__main__":
    main()
