from pathlib import Path

import streamlit as st
from download_data import get_frames, get_latest_ts_file
from object_tracking import (
    filter_parked_cars,
    filter_spurious_preds,
    get_northbound_southbound,
    get_preds,
    track_iou,
    write_video,
)

from landingai.predict import Predictor

VIDEO_CACHE_PATH = "cached_data"


def get_latest_traffic():
    Path(VIDEO_CACHE_PATH).mkdir(parents=True, exist_ok=True)
    get_latest_ts_file("vid.ts")
    frames = get_frames("vid.ts")
    predictor = Predictor(
        st.session_state["endpoint_id"],
        api_key=st.session_state["api_key"],
    )
    bboxes = get_preds(frames, predictor)
    tracks, all_idx_to_track = track_iou(bboxes)
    write_video(frames, bboxes, all_idx_to_track, "vid_out.mp4")
    tracks, parked = filter_parked_cars(tracks, len(frames))
    tracks, _ = filter_spurious_preds(tracks)
    northbound, southbound = get_northbound_southbound(tracks)
    st.video(open("vid_out.mp4", "rb").read())
    st.write(f"Northbound Traffic: **{len(northbound)}** cars per 10s")
    st.write(f"Southbound Traffic: **{len(southbound)}** cars per 10s")
    st.write(f"Parked Cars: **{parked}**")


st.title("Landing AI Traffic Counter")
button = st.button("Get Latest Traffic", on_click=get_latest_traffic)
st.divider()
