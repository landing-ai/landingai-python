from pathlib import Path

import streamlit as st

from landingai.pipeline.image_source import FrameSet, NetworkedCamera
from landingai.pipeline.postprocessing import get_class_counts
from landingai.predict import Predictor

VIDEO_CACHE_PATH = Path("cached_data")
VIDEO_CACHE_PATH.mkdir(exist_ok=True, parents=True)
VIDEO_CACHE_PATH = VIDEO_CACHE_PATH / "latest.mp4"
VIDEO_LEN_SEC = 10
FPS = 2
PLAYLIST_URL = (
    "https://live.hdontap.com/hls/hosb1/topanga_swellmagnet.stream/playlist.m3u8"
)


def get_latest_surfer_count():
    vid_src = NetworkedCamera(PLAYLIST_URL, fps=FPS)
    surfer_model = Predictor(
        st.session_state["endpoint_id"], api_key=st.session_state["api_key"]
    )

    frs = FrameSet()
    for i, frame in enumerate(vid_src):
        if i >= VIDEO_LEN_SEC * FPS:
            break
        frs.extend(frame)
    frs.run_predict(predictor=surfer_model).overlay_predictions()
    frs.save_video(str(VIDEO_CACHE_PATH), video_fps=FPS, image_src="overlay")
    surfers = (get_class_counts(frs)["surfer"]) / (VIDEO_LEN_SEC * FPS)
    st.video(open(VIDEO_CACHE_PATH, "rb").read())
    st.write(f"Surfer count: **{surfers}**")


st.title("Surfer Counter")
button = st.button("Get Topanga Beach Surfer Count", on_click=get_latest_surfer_count)
