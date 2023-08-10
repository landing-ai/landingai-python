import tempfile
from pathlib import Path

import streamlit as st
from streamlit_image_select import image_select

import landingai.pipeline as pl
from landingai import visualize
from landingai.postprocess import crop
from landingai.predict import OcrPredictor, Predictor
from landingai.st_utils import (
    get_api_key_or_use_default,
    render_api_config_form,
    setup_page,
)

setup_page(page_title="License Plate Detection and Recognition")

st.sidebar.title("Configuration")
api_key = "land_sk_aMemWbpd41yXnQ0tXvZMh59ISgRuKNRKjJEIUHnkiH32NBJAwf"
od_model_endpoint = "e001c156-5de0-43f3-9991-f19699b31202"

with st.sidebar:
    render_api_config_form()


def detect_license_plates(frames):
    bounding_boxes = []
    overlayed_frames = []

    predictor = Predictor(od_model_endpoint, api_key=api_key)
    od_predictor_bar = st.progress(0.0, text="Detecting license plates...")

    for i, frame in enumerate(frames):
        prediction = predictor.predict(frame)
        # store predictions in a list
        overlay = visualize.overlay_predictions(prediction, frame)
        bounding_boxes.append(prediction)
        overlayed_frames.append(overlay)
        od_predictor_bar.progress((i + 1) / len(frames), "Detecting license plates...")

    return bounding_boxes, overlayed_frames


def extract_frames(video):
    temp_dir = tempfile.mkdtemp()
    saved_video_file = Path(temp_dir) / video.name
    saved_video_file.write_bytes(video.read())
    video_source = pl.image_source.VideoFile(
        str(saved_video_file), samples_per_second=1
    )
    frames = []
    with st.spinner(text="Extracting frames from video file..."):
        frames.extend(frame_info.frames[0].image for frame_info in video_source)
    st.success("Frame Extraction Finished!")
    with st.expander("Preview extracted frames"):
        selected_img = image_select(
            label=f"Total {len(frames)} images",
            images=frames,
            key="preview_input_images",
            use_container_width=False,
        )
        st.image(selected_img)
    return frames

st.caption("Download below sample video file to try out the app or upload yours.")
st.video("https://drive.google.com/uc?id=16iwE7mcz9zHqKCw2ilx0QEwSCjDdXEW4")

if video := st.file_uploader("Upload a video file contains license plates to get started"):
    st.video(video)
    frames = extract_frames(video)
    # run prediction of frames
    bounding_boxes, overlayed_frames = detect_license_plates(frames)

    # show frames with overlayed bounding boxes
    for i, frame in enumerate(overlayed_frames):
        if len(bounding_boxes[i]) == 0:
            continue
        st.image(frame, width=800)

    cropped_imgs = [
        crop(bboxes, frame) for frame, bboxes in zip(frames, bounding_boxes)
    ]

    st.subheader(f"Found and cropped {len(cropped_imgs)} license plates below")
    # show 5 overlayed frames
    for i, cropped in enumerate(cropped_imgs):
        if len(cropped) == 0:
            continue
        for plate in cropped:
            st.image(plate)

    # run OCR
    # set staging OCR API key
    api_key = get_api_key_or_use_default()
    if not api_key:
        st.error("Please set your API key in the sidebar to run OCR.")
        st.stop()

    st.subheader(f"Run OCR on the above {len(cropped_imgs)} license plates")
    ocr_predictor = OcrPredictor(api_key=api_key)
    ocr_preds, overlayed_ocr = [], []
    ocr_predictor_bar = st.progress(0.0, text="Run OCR prediction...")
    for frame in cropped_imgs:
        for plate in frame:
            ocr_pred = ocr_predictor.predict(plate)
            ocr_preds.append(ocr_pred)
            overlay = visualize.overlay_predictions(ocr_pred, plate)
            overlayed_ocr.append(overlay)
            ocr_predictor_bar.progress(
                (i + 1) / len(cropped_imgs), "Running OCR prediction..."
            )

    for frame, ocr_pred in zip(overlayed_ocr, ocr_preds):
        if len(ocr_pred) == 0:
            continue
        st.image(frame)
        for text in ocr_pred:
            st.write(text.text)
