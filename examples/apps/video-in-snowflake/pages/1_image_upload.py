import logging
from pathlib import Path

import streamlit as st
from streamlit_image_select import image_select

from landingai.io import probe_video, sample_images_from_video
from landingai.storage.snowflake import save_remote_file_to_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_MEDIA_CACHE_DIR = Path("examples/output/streamlit")


def reset_states():
    st.session_state["image_paths_for_upload"] = []


def upload_images_to_landinglens():
    st.write("Uploading images to LandingLens...")
    st.write("Done!")


def is_landing_credentials_set():
    return (
        st.session_state.get("endpoint_id")
        and st.session_state.get("api_key")
        and st.session_state.get("api_secret")
        and st.session_state.get("snow_config")
    )


def preview_images(video_file_path, samples_per_second):
    images = sample_images_from_video(
        video_file_path, _MEDIA_CACHE_DIR, samples_per_second
    )
    st.session_state["image_paths_for_upload"] = images


st.divider()
st.subheader("Upload images for training")
option = st.radio("File Storage", options=["Local", "Snowflake"], index=1)
col1, col2 = st.columns(2)
with col1:
    file_path = st.text_input(
        "Video File Link",
        key="file_path_for_upload",
        value="",
        on_change=reset_states,
    )
    local_video_file_path_for_upload = None
    if option == "Local":
        local_video_file_path_for_upload = file_path
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
            local_video_file_path_for_upload = str(_MEDIA_CACHE_DIR / file_path)
            logging.info(f"Found {local_video_file_path_for_upload} in local cache")
        elif st.button("Confirm", key="download_to_local"):
            snow_credential, snow_config = st.session_state["snow_config"]
            local_video_file_path_for_upload = save_remote_file_to_local(
                remote_filename=file_path,
                stage_name=stage_name,
                local_output=_MEDIA_CACHE_DIR,
                credential=snow_credential,
                connection_config=snow_config,
            )
            local_video_file_path_for_upload = str(local_video_file_path_for_upload)
            logging.info(
                f"Downloaded {file_path} to local cache: {local_video_file_path_for_upload}"
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
        key="samples_per_second_for_upload",
    )


if local_video_file_path_for_upload:
    logging.info(
        f"Using video file {local_video_file_path_for_upload} with {samples_per_second}"
    )
    total_frames, sample_size, video_length_seconds = probe_video(
        local_video_file_path_for_upload, samples_per_second
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Video length(s)", video_length_seconds)
    with col2:
        st.metric("Total number of frames", total_frames)
    with col3:
        st.metric("Images after sampling", sample_size)

    if not is_landing_credentials_set():
        st.error(
            "Please go to the config page and enter your CloudInference endpoint ID first."
        )
        st.stop()

    preview = st.button(
        "Sample images from the video file",
        on_click=preview_images,
        args=(local_video_file_path_for_upload, samples_per_second),
        type="secondary",
        key="sample_images_from_video_for_upload",
    )
    image_paths = st.session_state.get("image_paths_for_upload", [])
    if preview or image_paths:
        with st.expander("Preview input images"):
            img_path = image_select(
                label=f"Total {len(image_paths)} sampled images",
                images=image_paths,
                captions=[f"Frame {Path(p).stem}" for p in image_paths],
                use_container_width=False,
            )
            st.image(img_path, caption=f"Frame {Path(img_path).stem}")

    st.button("Upload", on_click=upload_images_to_landinglens, type="secondary")
st.divider()
