import logging
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple, Union

import cv2
import requests

_LOGGER = logging.getLogger(__name__)


# TODO: support output type stream
def read_file(url: str) -> bytes:
    """Read bytes from a URL.
    Typically, the URL is a presigned URL (for example, from Amazon S3 or Snowflake) that points to a video or image file.
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


def probe_video(video_file: str, samples_per_second: float) -> Tuple[int, int, float]:
    """Probe a video file to get some metadata before sampling images.

    Parameters
    ----------
    video_file: The local path to the video file
    samples_per_second: Number of images to sample per second

    Returns
    -------
    A tuple of three values

        - The total number of frames,
        - The number of frames to sample,
        - The video length in seconds.
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
) -> List[str]:
    """Sample images from a video file.

    Parameters
    ----------
    video_file: The local path to the video file
    output_dir: The local directory path that stores the sampled images
    samples_per_second: The number of images to sample per second

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


def read_from_notebook_webcam(webcam_source: Union[str, int] = 0) -> Callable[[], str]:
    # Define function to acquire images either directly from the local webcam (i.e. jupyter notebook)or from the web browser (i.e. collab)
    local_cache_dir = Path(tempfile.mkdtemp())
    filename = str(local_cache_dir / "photo.jpg")
    # Detect if we are running on Google's colab
    try:
        from base64 import b64decode

        from google.colab.output import eval_js  # type: ignore
        from IPython.display import Javascript, display

        def take_photo() -> str:
            quality = 0.8
            js = Javascript(
                """
            async function takePhoto(quality) {
                const div = document.createElement('div');
                const capture = document.createElement('button');
                capture.textContent = 'Capture';
                div.appendChild(capture);

                const video = document.createElement('video');
                video.style.display = 'block';
                const stream = await navigator.mediaDevices.getUserMedia({video: true});

                document.body.appendChild(div);
                div.appendChild(video);
                video.srcObject = stream;
                await video.play();

                // Resize the output to fit the video element.
                google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

                // Wait for Capture to be clicked.
                await new Promise((resolve) => capture.onclick = resolve);

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                return canvas.toDataURL('image/jpeg', quality);
            }
            """
            )
            display(js)
            data = eval_js("takePhoto({})".format(quality))
            binary = b64decode(data.split(",")[1])
            with open(filename, "wb") as f:
                f.write(binary)
                return filename

    except ModuleNotFoundError:
        # Capture image from local webcam using OpenCV
        import cv2

        def take_photo() -> str:
            cam = cv2.VideoCapture(webcam_source)
            cv2.namedWindow("Press space to take photo")
            cv2.startWindowThread()
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    exit()
                cv2.imshow("Press space to take photo", frame)
                k = cv2.waitKey(1)
                if k % 256 == 32:
                    # SPACE pressed
                    cv2.imwrite(filename, frame)
                    break
            cam.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return filename

    return take_photo
