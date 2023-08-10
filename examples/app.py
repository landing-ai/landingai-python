import streamlit as st
import requests
import cv2
import numpy as np
import time
from landingai.postprocess import crop
from landingai.predict import Predictor, OcrPredictor
from landingai import visualize
import landingai.pipeline as pl
from pathlib import Path
import tempfile
import PIL.Image


video = st.file_uploader("File Uploader")
if video:
    st.video(video)
    temp_dir = tempfile.mkdtemp()
    saved_video_file = Path(temp_dir) / video.name
    saved_video_file.write_bytes(video.read())
    video_source = pl.image_source.VideoFile(str(saved_video_file), samples_per_second=1)
    frames = []
    for frame in video_source:
        frames.append(frame.frames[0].image)
    # Show frames
    for frame in frames:
        st.image(frame, width=200)


# run prediction of frames 

    def detect_license_plates(frames):
        bounding_boxes = []
        overlayed_frames = []
        api_key = "land_sk_OdafnFLV340HT1eCdvm3Z4X3Xev8VP58iAhfqh6hAdnORL9ySq"
        model_endpoint = "972bcd20-31fc-4537-96f4-8b92e3a91408"
        predictor = Predictor(model_endpoint, api_key=api_key)
        
        for frame in frames:
            prediction = predictor.predict(frame)
            # store predictions in a list
            # convert to rgb
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = visualize.overlay_predictions(prediction, frame)
            bounding_boxes.append(prediction)
            overlayed_frames.append(overlay)

        return bounding_boxes, overlayed_frames

    bounding_boxes, overlayed_frames = detect_license_plates(frames)

    # show 5 overlayed frames
    for i, frame in enumerate(overlayed_frames):
        if len(bounding_boxes[i]) == 0:
            continue
        st.image(frame, width=800)


# cropping 

# cropping the license plate
    cropped_imgs = []
    for frame, bboxes in zip(frames, bounding_boxes):
        cropped_imgs.append(crop(bboxes, frame))

    print(len(cropped_imgs))
    # show 5 overlayed frames
    for i, cropped in enumerate(cropped_imgs):
        if len(cropped) == 0:
            continue
        for plate in cropped:
            st.image(plate)


# run OCR


# set staging OCR API key 

    ocr_predictor = OcrPredictor(api_key='land_sk_EkHnd6IDQvRVpgRcA3xCcUDAjDNqogs8Z3EKidTRctlZogIZwp')

    ocr_preds = []
    overlayed_ocr = []
    # print(cropped_imgs[0])
    for frame in cropped_imgs:
        for plate in frame:
            ocr_pred = ocr_predictor.predict(plate)
            ocr_preds.append(ocr_pred)
            overlay = visualize.overlay_predictions(ocr_pred, plate)
            overlayed_ocr.append(overlay)
    # print(ocr_preds)
    for frame, ocr_pred in zip(overlayed_ocr, ocr_preds):
        if len(ocr_pred) == 0:
            continue
        st.image(frame)
        for text in ocr_pred:
            st.write(text.text)