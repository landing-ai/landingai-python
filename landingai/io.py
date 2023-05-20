from pathlib import Path

import cv2


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
    cv2.destroyAllWindows()
    return output
