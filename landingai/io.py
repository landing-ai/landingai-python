import logging
from pathlib import Path

import cv2
import requests

_LOGGER = logging.getLogger(__name__)


# TODO: support output type stream
def read_file(url: str) -> bytes:
    """Read bytes from a url.
    Typically, the url is a presigned url (e.g. from s3, snowflake) that points to a video, image file.
    """
    response = requests.get(url)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        reason = f"{e.response.text} (status code: {e.response.status_code})"
        msg_prefix = f"Failed to read from url ({url}) due to {reason}"
        if response.status_code == 403:
            error_msg = f"{msg_prefix}. Please double check the url is not expired and it's well-formed."
            raise ValueError(error_msg) from e
        elif response.status_code == 404:
            raise FileNotFoundError(
                f"{msg_prefix}. Please double check the file exists and the url is well-formed."
            ) from e
        else:
            error_msg = f"{msg_prefix}. Please try again later or reach out to us via our LandingAI platform."
            raise ValueError(error_msg) from e
    if response.status_code >= 300:
        raise ValueError(
            f"Failed to read from url ({url}) due to {response.text} (status code: {response.status_code})"
        )
    _LOGGER.info(
        f"Received content with length {len(response.content)} and type {response.headers.get('Content-Type')}"
    )
    return response.content


def probe_video(video_file: str, samples_per_second: float) -> tuple[int, int, float]:
    """Probe a video file to get some metadata before sampling images.

    Parameters
    ----------
    video_file: the local path to the video file
    samples_per_second: number of images to sample per second

    Returns
    -------
    A tuple of three values:
    1. the total number of frames,
    2. the number of frames to sample,
    3. the video length in seconds.
    """
    if not Path(video_file).exists():
        raise FileNotFoundError(f"Video file {video_file} does not exist.")
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_seconds = total_frames / fps
    sample_size = int(video_length_seconds * samples_per_second)
    cap.release()
    return (total_frames, sample_size, video_length_seconds)


def sample_images_from_video(
    video_file: str, output_dir: Path, samples_per_second: float = 1
) -> list[str]:
    """Sample images from a video file.

    Parameters
    ----------
    video_file: the local path to the video file
    output_dir: the local directory path that stores the sampled images
    samples_per_second: number of images to sample per second

    Returns
    -------
    a list of local file paths to the sampled images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames, sample_size, _ = probe_video(video_file, samples_per_second)
    # Calculate the frame interval based on the desired frame rate
    sample_interval = int(total_frames / sample_size)
    frame_count = 0
    output = []
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        # Check if end of video file
        if not ret:
            break
        # Check if the current frame should be saved
        if frame_count % sample_interval == 0:
            output_file_path = str(output_dir / f"{frame_count}.jpg")
            cv2.imwrite(output_file_path, frame)
            output.append(output_file_path)
        frame_count += 1
    cap.release()
    return output
