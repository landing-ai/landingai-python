import datetime
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import streamlit as st
from streamlit_image_select import image_select

from landingai.common import Prediction
from landingai.io import probe_video, sample_images_from_video
from landingai.postprocess import class_counts, class_map
from landingai.predict import Predictor
from landingai.storage.snowflake import save_remote_file_to_local
from landingai.visualize import overlay_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_MEDIA_CACHE_DIR = Path("examples/output/streamlit")
_VIDEO_OUTPUT_PAHT = str(_MEDIA_CACHE_DIR / "output.mp4")
_MAX_VIDEO_LENGTH_SECONDS = 15


@dataclass(frozen=True)
class VideoFrameInferenceResult:
    local_image_path: Path
    image: np.ndarray
    predictions: list[Prediction]

    @cached_property
    def frame_index(self) -> int:
        return int(self.local_image_path.stem)

    @cached_property
    def class_counts(self) -> dict[int, (int, str)]:
        return class_counts(self.predictions)

    @cached_property
    def image_with_predictions(self) -> PIL.Image.Image:
        return overlay_predictions(
            self.predictions, self.image, options={"label_type": "t-label"}
        )


@dataclass(frozen=True)
class VideoInferenceResult:
    local_video_file_path: Path
    frame_predictions: list[VideoFrameInferenceResult]

    @cached_property
    def class_map(self) -> dict[int, str]:
        all_predictions = []
        for res in self.frame_predictions:
            all_predictions.extend(res.predictions)
        return class_map(all_predictions)

    @cached_property
    def fps(self) -> int:
        cap = cv2.VideoCapture(str(self.local_video_file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        predictions = self.frame_predictions
        frame_indices = [int(pred.frame_index) for pred in predictions]
        images = [pred.image for pred in predictions]
        image_paths = [str(pred.local_image_path) for pred in predictions]
        class_counts_all = {
            class_name: [0] * len(images) for class_name in self.class_map.values()
        }
        for i, res in enumerate(predictions):
            for count, class_name in res.class_counts.values():
                class_counts_all[class_name][i] = count
        occurence_time = [
            str(datetime.timedelta(seconds=int(i / self.fps))) for i in frame_indices
        ]
        data = {
            "frame_index": frame_indices,
            "occurence_time": occurence_time,
            "image_path": image_paths,
        }
        data.update(class_counts_all)
        return pd.DataFrame(data)

    @cached_property
    def frame_index_result_map(self):
        return {f.frame_index: f for f in self.frame_predictions}

    def get_frame_inference_result(self, frame_idx: int) -> VideoFrameInferenceResult:
        result = self.frame_index_result_map.get(frame_idx)
        assert (
            result is not None
        ), f"Development bug, frame index {frame_idx} not found, available frame indices: {self.frame_index_result_map.keys()}"
        return result


def preview_images(video_file_path, samples_per_second):
    images = sample_images_from_video(
        video_file_path, _MEDIA_CACHE_DIR, samples_per_second
    )
    st.session_state["image_paths"] = images


def bulk_inference(
    image_paths: list[str], video_file_path: str
) -> VideoInferenceResult:
    endpoint_id = st.session_state["endpoint_id"]
    api_key = st.session_state["api_key"]
    api_secret = st.session_state["api_secret"]
    predictor = Predictor(endpoint_id, api_key=api_key, api_secret=api_secret)
    images = [np.asarray(PIL.Image.open(p)) for p in image_paths]
    pbar = st.progress(0, text="Running inferences...")
    result = []
    percent_complete = 1 / len(images)
    for i, (img, img_path) in enumerate(zip(images, image_paths)):
        preds = predictor.predict(img)
        result.append(
            VideoFrameInferenceResult(
                local_image_path=Path(img_path),
                image=img,
                predictions=preds,
            )
        )
        percent_progress = percent_complete * (i + 1)
        text = f"Running inferences... {percent_progress:.1%} completed"
        pbar.progress(percent_progress, text=text)
    video_result = VideoInferenceResult(Path(video_file_path), result)
    st.session_state["result"] = video_result
    return video_result


def make_video(result: VideoInferenceResult) -> str:
    assert result is not None, "Development bug, result should not be None"
    # prev_result = st.session_state["result"]
    # if result == prev_result:
    #     return video_file_path
    video_file_path = _VIDEO_OUTPUT_PAHT
    # All images should have the same shape as it's from the same video file
    img_shape = result.frame_predictions[0].image.shape[:2][::-1]
    total_frames = len(result.frame_predictions)
    frame_rate = int(total_frames / _MAX_VIDEO_LENGTH_SECONDS)
    # TODO: cap the frame_rate to a reasonable value
    # H264 is preferred, see https://discuss.streamlit.io/t/st-video-doesnt-show-opencv-generated-mp4/3193/4
    video = cv2.VideoWriter(
        video_file_path, cv2.VideoWriter_fourcc(*"H264"), frame_rate, img_shape
    )
    for res in result.frame_predictions:
        img_with_preds = overlay_predictions(
            res.predictions, res.image, options={"label_type": "t-label"}
        )
        img_np = np.asarray(img_with_preds)
        video.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    video.release()
    logging.info(f"Video is saved to {video_file_path} with resolution {img_shape}")
    return video_file_path


def result_overview(reuslt: VideoInferenceResult):
    result_df = reuslt.dataframe
    classes = list(reuslt.class_map.values())
    fig_all = px.bar(
        result_df,
        x="occurence_time",
        y=classes,
        title="Detected Cars Over Time",
        labels={
            "occurence_time": "Video time",
            "value": "Number of detected cars",
        },
    )
    st.plotly_chart(fig_all, use_container_width=True)


def analyze_video_frame_result(reuslt: VideoInferenceResult):
    result_df = reuslt.dataframe
    st.subheader("Pick a frame to see the details")
    frame_idx_map = dict(zip(result_df["occurence_time"], result_df["frame_index"]))
    options = result_df["occurence_time"].tolist()
    # st.session_state["selected_time_frame_idx"] = options[0]
    selected_time = st.select_slider(
        label="Select a time of the video to see the details",
        options=options,
        # value=options[0],
        key="selected_time_frame_idx",
    )
    selected_frame_idx = frame_idx_map[selected_time]
    frame_inference_result: VideoFrameInferenceResult = (
        video_result.get_frame_inference_result(selected_frame_idx)
    )
    st.image(
        frame_inference_result.image_with_predictions,
        caption=f"Selected frame at {selected_time}",
    )
    cls_counts = frame_inference_result.class_counts.values()
    fig_per_frame = px.bar(
        x=[val[1] for val in cls_counts],
        y=[val[0] for val in cls_counts],
        title="Detected Cars at a given time",
        labels={
            "x": "Car type",
            "y": "Number of detected cars",
        },
    )
    st.plotly_chart(fig_per_frame)


def is_landing_credentials_set():
    return (
        st.session_state.get("endpoint_id")
        and st.session_state.get("api_key")
        and st.session_state.get("api_secret")
        and st.session_state.get("snow_config")
    )


def reset_states():
    st.session_state["image_paths"] = []
    st.session_state["result"] = []


st.divider()
st.subheader("Run inferences")
option = st.radio("File Storage", options=["Local", "Snowflake"], index=1)
col1, col2 = st.columns(2)
with col1:
    file_path = st.text_input(
        "Video File Link",
        key="snowflake_warehouse",
        value="",
        on_change=reset_states,
    )
    local_video_file_path = None
    if option == "Local":
        local_video_file_path = file_path
    elif option == "Snowflake":
        if not is_landing_credentials_set():
            st.error(
                "Please go to the config page and set up your Snowflake credentials first."
            )
            st.stop()
        stage_name = st.text_input(
            "Snowflake Stage Name", key="stage_name", value="VIDEO_FILES_STAGE"
        )
        if file_path and (_MEDIA_CACHE_DIR / file_path).exists():
            local_video_file_path = str(_MEDIA_CACHE_DIR / file_path)
            logging.info(f"Found {local_video_file_path} in local cache")
        elif st.button("Confirm", key="download_to_local"):
            snow_credential, snow_config = st.session_state["snow_config"]
            local_video_file_path = save_remote_file_to_local(
                remote_filename=file_path,
                stage_name=stage_name,
                local_output=_MEDIA_CACHE_DIR,
                credential=snow_credential,
                connection_config=snow_config,
            )
            local_video_file_path = str(local_video_file_path)
            logging.info(
                f"Downloaded {file_path} to local cache: {local_video_file_path}"
            )
    else:
        raise ValueError(f"Unknown option {option}")
with col2:
    samples_per_second = st.slider(
        "Samples per second",
        min_value=0.1,
        max_value=20.0,
        step=0.1,
        value=1.0,
        key="samples_per_second",
    )


if local_video_file_path:
    total_frames, sample_size, video_length_seconds = probe_video(
        local_video_file_path, samples_per_second
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Video length(s)", video_length_seconds)
    with col2:
        st.metric("Total number of frames", total_frames)
    with col3:
        st.metric("Images after sampling", sample_size)

    # TODO: limit max samples size
    if not is_landing_credentials_set():
        st.error(
            "Please go to the config page and enter your CloudInference endpoint ID first."
        )
        st.stop()

    preview = st.button(
        "Sample images from the video file",
        on_click=preview_images,
        args=(local_video_file_path, samples_per_second),
        type="secondary",
    )
    image_paths = st.session_state.get("image_paths", [])
    if preview or image_paths:
        with st.expander("Preview input images"):
            img_path = image_select(
                label=f"Total {len(image_paths)} sampled images",
                images=image_paths,
                captions=[f"Frame {Path(p).stem}" for p in image_paths],
                use_container_width=False,
            )
            st.image(img_path, caption=f"Frame {Path(img_path).stem}")

        st.divider()
        request_to_run_inference = st.button(
            "Run inferences",
            on_click=bulk_inference,
            kwargs={
                "image_paths": image_paths,
                "video_file_path": local_video_file_path,
            },
            type="secondary",
        )
        video_result = st.session_state.get("result", None)
        if not video_result:
            logging.info("No inference result found")
            st.stop()
        st.info(
            f"Inference finished! Received inference result for {len(video_result.frame_predictions)} images."
        )
        tab1, tab2, tab3 = st.tabs(["Result by frame", "Video stats", "Video output"])
        with tab1:
            analyze_video_frame_result(video_result)
        with tab2:
            result_overview(video_result)
        with tab3:
            if st.button("Generate video on inference result"):
                make_video(video_result)
                st.video(_VIDEO_OUTPUT_PAHT, format="video/mp4")
st.divider()
