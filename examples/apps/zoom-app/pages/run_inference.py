import psutil
import numpy as np
import streamlit as st

from sys import platform
from landingai.pipeline.image_source import NetworkedCamera
from landingai.predict import Predictor


is_win = platform == "win32"


if "api_key" in st.session_state and "endpoint_id" in st.session_state:
    webcam_on = False
    if is_win:
        from webcam_detect import WebcamDetect

        webcam_detect = WebcamDetect()
        webcam_on = webcam_detect.is_active_app("zoom")

    if (
        len([p for p in psutil.process_iter() if "zoom" in p.name().lower()]) > 0
        and webcam_on
    ):
        model = Predictor(
            st.session_state["endpoint_id"], api_key=st.session_state["api_key"]
        )
        video_src = NetworkedCamera(0, fps=1)
        placeholder = st.empty()

        for frame in video_src:
            frame.run_predict(model).overlay_predictions()
            if len(frame.frames) > 0:
                frame_with_pred = frame.frames[-1].other_images["overlay"]
                placeholder.empty()
                with placeholder.container():
                    st.image(np.array(frame_with_pred))
    else:
        st.warning(
            "App will not run if Zoom is not running or if your webcam is off (for Windows). To run, start Zoom, turn on your webcam and click on 'run inference' again"
        )
else:
    st.warning("Please enter your API Key and Endpoint ID in the sidebar.")
