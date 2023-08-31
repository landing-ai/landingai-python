"""Operations (create, probe, save, etc) around images, videos from various sources (webcam, video, RTSP stream etc)."""

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image


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
    samples_per_second: The number of images to sample per second. If set to zero, it disables sampling
    Returns
    -------
    a list of local file paths to the sampled images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if samples_per_second != 0:
        total_frames, sample_size, _ = probe_video(video_file, samples_per_second)
        # Calculate the frame interval based on the desired frame rate
        sample_interval = int(total_frames / sample_size)
    else:
        sample_interval = 1
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


def take_photo_from_webcam(webcam_source: Union[str, int] = 0) -> PIL.Image.Image:
    """Open a Window and allow users to take a photo from the webcam.
    It supports running on both local environment and Google's colab notebooks.
    For Google colab, it will open a window to allow users to take a photo from the webcam by pressing the "Capture" button.
    For local environment, it will open a window to allow users to take a photo from the webcam by pressing the "space" button.
    """
    try:
        # Running on Google's colab or local Jupyter notebook
        from base64 import b64decode

        from google.colab.output import eval_js  # type: ignore
        from IPython.display import Javascript, display

        # Define function to acquire images either directly from the local webcam (i.e. jupyter notebook)or from the web browser (i.e. collab)
        def take_photo() -> PIL.Image.Image:
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
            return PIL.Image.open(io.BytesIO(binary))

    except ModuleNotFoundError:
        # Capture image from local webcam using OpenCV
        import cv2

        def take_photo() -> PIL.Image.Image:
            cam = cv2.VideoCapture(webcam_source)
            cv2.namedWindow("Press space to take photo")
            cv2.startWindowThread()
            captured_frame: Optional[np.ndarray] = None
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    exit()
                cv2.imshow("Press space to take photo", frame)
                k = cv2.waitKey(1)
                if k % 256 == 32:
                    # SPACE pressed
                    captured_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    break
            cam.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return PIL.Image.fromarray(captured_frame)

    return take_photo()
