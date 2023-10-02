"""A module that provides a set of abstractions and APIs for reading images from different sources."""

import glob
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Type
from typing import Iterator as IteratorType
from typing import List, Optional, Union, Tuple
import warnings

import cv2
from PIL import ImageGrab
import numpy as np
from pydantic import BaseModel, PrivateAttr

from landingai.image_source_ops import sample_images_from_video
from landingai.pipeline.frameset import Frame
from landingai.storage.data_access import fetch_from_uri


_LOGGER = logging.getLogger(__name__)
_DEFAULT_FPS = 30


class ImageSourceBase(Iterator):
    """The base class for all image sources."""

    def close(self) -> None:
        """Free up any resource used by the image source"""
        pass

    def __next__(self) -> Frame:
        raise NotImplementedError()

    def __enter__(self) -> "ImageSourceBase":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False


class ImageFolder(ImageSourceBase):
    """
    The `ImageFolder` class is an image source that reads images from a folder path.

    Example 1:
    ```python
    folder = ImageFolder("/home/user/images")
    for image_batch in folder:
        print(image_batch[0].image.size)
    ```

    Example 2:
    ```python
    # Read all jpg files in the folder (including nested files)
    folder = ImageFolder(glob_pattern="/home/user/images/**/*.jpg")
    for image_batch in folder:
        print(image_batch[0].image.size)
    ```
    """

    def __init__(
        self,
        source: Union[Path, str, List[str], None] = None,
        glob_pattern: Union[str, List[str], None] = None,
    ) -> None:
        """Constructor for ImageFolder.

        Parameters
        ----------
        source
            A list of file paths or the path to the folder path that contains the images.
            A folder path can be either an absolute or relative folder path, in `str` or `Path` type. E.g. "/home/user/images".
            If you provide a folder path, all the files directly within the folder will be read (including non-image files).
            Nested files and sub-directories will be ignored.
            Consider using `glob_pattern` if you need to:
              1. filter out unwanted files, e.g. your folder has both image and non-image files
              2. read nested image files, e.g. `/home/user/images/**/*.jpg`.
            The ordering of images is based on the file name alphabetically if source is a folder path.
            If source is a list of files, the order of the input files is preserved.
            Currently only local file paths are supported.
        glob_pattern
            One or more python glob pattern(s) to grab only wanted files in the folder. E.g. "/home/user/images/*.jpg".
            NOTE: If `glob_pattern` is provided, the `source` parameter is ignored.
            For more information about glob pattern, see https://docs.python.org/3/library/glob.html

        """
        self._source = source
        self._image_paths: List[str] = []

        if source is None and glob_pattern is None:
            raise ValueError("Either 'source' or 'glob_pattern' must be provided.")
        if glob_pattern is not None:
            if isinstance(glob_pattern, str):
                glob_pattern = [glob_pattern]
            for pattern in glob_pattern:
                self._image_paths.extend(list(glob.glob(pattern, recursive=True)))
            self._image_paths.sort()
        elif isinstance(source, list):
            self._image_paths = source
        else:
            assert isinstance(source, str) or isinstance(source, Path)
            p = Path(source)
            if not p.exists():
                raise ValueError(f"Path '{p}' does not exist.")
            self._image_paths = [str(x) for x in p.glob("*") if x.is_file()]
            self._image_paths.sort()

    def __iter__(self) -> IteratorType[Frame]:
        for img_path in self._image_paths:
            meta = {"image_path": img_path}
            yield Frame.from_image(str(img_path), metadata=meta)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __repr__(self) -> str:
        return str(self._image_paths)

    @property
    def image_paths(self) -> List[str]:
        """Returns a list of image paths."""
        return self._image_paths


class VideoFile(ImageSourceBase):
    """
    The `VideoFile` class is an image source that samples frames from a video file.

    Example:
    ```python
        import landingai.pipeline as pl

        img_src = pl.image_source.VideoFile("sample_images/surfers.mp4", samples_per_second=1)
        frs = pl.FrameSet()
        for i,frame in enumerate(img_src):
            if i>=3: # Fetch only 3 frames
                break
            frs.extend(
                frame.run_predict(predictor=surfer_model)
                .overlay_predictions()
            )
        print(pl.postprocessing.get_class_counts(frs))
    ```
    """

    def __init__(self, uri: str, samples_per_second: float = 1) -> None:
        """Constructor for VideoFile.

        Parameters
        ----------
        uri : str
            URI to the video file. This could be a local file or a URL that serves the video file in bytes.
        samples_per_second : float, optional
            The number of images to sample per second (by default 1). If set to zero, it disables sampling
        """
        self._video_file = str(fetch_from_uri(uri))
        self._local_cache_dir = Path(tempfile.mkdtemp())
        self._samples_per_second = samples_per_second
        cap = cv2.VideoCapture(self._video_file)
        self._src_fps = cap.get(cv2.CAP_PROP_FPS)
        if self._src_fps == 0:
            _LOGGER.warning(
                f"Could not read FPS from video file ({self._video_file}). Set src FPS to {_DEFAULT_FPS}"
            )
            self._src_fps = _DEFAULT_FPS
        self._src_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Compute video properties (i.e. the target number of samples and FPS based on the user provided `samples_per_second`)
        if self._samples_per_second != 0:
            # Adjust the video frames and fps based on sampling rate
            self._target_total_frames = int(
                self._src_total_frames * self._samples_per_second / self._src_fps
            )
            self._target_fps = self._samples_per_second
        else:
            self._target_total_frames = self._src_total_frames
            self._target_fps = self._src_fps

    def properties(self) -> Tuple[float, int, float, int]:
        """Return properties of the of the source file and the resulting FrameSet

        Returns
        -------
        Tuple[float, int, float, int]
            Properties:
            0. Source file FPS (frames per second)
            1. Source file total number of frames
            2. Resulting FPS after applying sampling rate
            3. Number of frames after applying sampling rate
        """
        return (
            self._src_fps,
            self._src_total_frames,
            self._target_fps,
            self._target_total_frames,
        )

    def __iter__(self) -> IteratorType[Frame]:
        for img_path in sample_images_from_video(
            self._video_file, self._local_cache_dir, self._samples_per_second
        ):
            yield Frame.from_image(img_path)

    def __del__(self) -> None:
        if os.path.exists(self._local_cache_dir):
            # TODO: deprecate the __del__ method in future versions.
            warnings.warn(
                "VideoFile object should be closed explicitly with close(), or using 'with' statement."
            )
            self.close()

    def close(self) -> None:
        shutil.rmtree(self._local_cache_dir)


# openCV's default VideoCapture cannot drop frames so if the CPU is overloaded the stream will start to lag behind realtime.
# This class creates a treaded capture implementation that can stay up to date wit the stream and decodes frames only on demand
class NetworkedCamera(BaseModel, ImageSourceBase):
    """The NetworkCamera class can connect to RTSP and other live video sources in order to grab frames. The main concern is to be able to consume frames at the source speed and drop them as needed to ensure the application allday gets the lastes frame"""

    stream_url: str
    motion_detection_threshold: int
    capture_interval: Union[float, None] = None
    previous_frame: Union[Frame, None] = None
    _last_capture_time: datetime = PrivateAttr()
    _cap: Any = PrivateAttr()  # cv2.VideoCapture
    _t: Any = PrivateAttr()  # threading.Thread
    _t_lock: Any = PrivateAttr()  # threading.Lock
    _t_running: bool = PrivateAttr()

    def __init__(
        self,
        stream_url: Union[str, int],
        motion_detection_threshold: int = 0,
        capture_interval: Optional[float] = None,
        fps: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        stream_url : url to video source, or a number indicating the webcam index
        motion_detection_threshold : If set to zero then motion detections is disabled. Any other value (0-100) will make the camera drop all images that don't have significant changes
        capture_interval : Number of seconds to wait in between frames. If set to None, the NetworkedCamera will acquire images as fast as the source permits.
        fps: Capture speed in frames per second. If set to None, the NetworkedCamera will acquire images as fast as the source permits.
        """
        if fps is not None and capture_interval is not None:
            raise ValueError(
                "The fps and capture_interval arguments cannot be set at the same time"
            )
        elif fps is not None:
            capture_interval = 1 / fps

        if capture_interval is not None and capture_interval < 1 / 30:
            raise ValueError(
                "The resulting fps cannot be more than 30 frames per second"
            )
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            cap.release()
            raise Exception(f"Could not open stream ({stream_url})")
        cap.set(
            cv2.CAP_PROP_BUFFERSIZE, 2
        )  # Limit buffering to 2 frames in order to avoid backlog (i.e. lag)

        super().__init__(
            stream_url=stream_url,
            motion_detection_threshold=motion_detection_threshold,
            capture_interval=capture_interval,
        )
        self._last_capture_time = datetime.now()
        self._cap = cap
        self._t_lock = threading.Lock()
        self._t = threading.Thread(target=self._reader)
        self._t.daemon = True
        self._t_running = False
        self._t.start()

    def __del__(self) -> None:
        if self._t_running:
            # TODO: deprecate the __del__ method in future versions.
            warnings.warn(
                "NetworkedCamera object should be closed explicitly with close(), or using with statement."
            )
            self.close()

    def close(self) -> None:
        self._t_running = False
        self._t.join(timeout=10)
        self._cap.release()

    # grab frames as soon as they are available
    def _reader(self) -> None:
        self._t_running = True
        # inter_frame_interval = 1 / self._cap.get(cv2.CAP_PROP_FPS)  # Get the source's framerate (FPS = 1/X)
        inter_frame_interval = (
            1 / 30
        )  # Some sources miss report framerate so we use a conservative number
        while self._t_running:
            with self._t_lock:
                ret = self._cap.grab()  # non-blocking call
                if not ret:
                    raise Exception(f"Connection to camera broken ({self.stream_url})")
            time.sleep(inter_frame_interval)  # Limit acquisition speed

    # retrieve latest frame
    def get_latest_frame(self) -> Optional["Frame"]:
        """Return the most up to date frame by dropping all by the latest frame. This function is blocking"""
        if self.capture_interval is not None:
            t = datetime.now()
            delta = (t - self._last_capture_time).total_seconds()
            if delta <= self.capture_interval:
                time.sleep(self.capture_interval - delta)
        with self._t_lock:
            ret, frame = self._cap.retrieve()  # non-blocking call
            if not ret:
                raise Exception(f"Connection to camera broken ({self.stream_url})")
        self._last_capture_time = datetime.now()
        if self.motion_detection_threshold > 0:
            if self._detect_motion(frame):
                return Frame.from_array(frame)
            else:
                return None  # Empty frame

        return Frame.from_array(frame)

    def _detect_motion(self, frame: np.ndarray) -> bool:  # TODO Needs test cases
        """ """
        # Prepare image; grayscale and blur
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        if self.previous_frame is None:
            # Save the result for the next invocation
            self.previous_frame = prepared_frame
            return True  # First frame; there is no previous one yet

        # calculate difference and update previous frame TODO: don't assume the processed image is cached
        diff_frame = cv2.absdiff(src1=self.previous_frame, src2=prepared_frame)
        # Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(
            src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY
        )[1]
        change_percentage = (
            100 * cv2.countNonZero(thresh_frame) / (frame.shape[0] * frame.shape[1])
        )
        # print(f"Image change {change_percentage:.2f}%")
        if change_percentage > self.motion_detection_threshold:
            self.previous_frame = prepared_frame
            return True
        return False

    # Make the class iterable. TODO Needs test cases
    def __iter__(self) -> Any:
        return self

    def __next__(self) -> "Frame":
        latest_frame = self.get_latest_frame()
        while latest_frame is None:
            # If motion detection is set, we may get empty frames.
            # We should not yield then, though: we should wait for the next frame
            # until we get a real Frame.
            latest_frame = self.get_latest_frame()
        return latest_frame

    class Config:
        arbitrary_types_allowed = True


class Webcam(NetworkedCamera):
    """
    The Webcam class can connect to a local webcam in order to iterate over captured frames.

    This leverages the NetworkedCamera implementations, with the constructor
    receiving the webcam ID to be sent to OpenCV (instead of a stream URL).
    Note that it doesn't work with Collab or remote Jupyter notebooks (yet). In this case,
    use `landingai.image_source_ops.take_photo_from_webcam()` instead.
    """

    def __init__(
        self,
        webcam_source: int = 0,
        motion_detection_threshold: int = 0,
        capture_interval: Optional[float] = None,
        fps: Optional[int] = None,
    ) -> None:
        super().__init__(
            webcam_source,
            motion_detection_threshold=motion_detection_threshold,
            capture_interval=capture_interval,
            fps=fps,
        )


class Screenshot(ImageSourceBase):
    """
    Take a screenshot from the screen as an image source.

    The screenshot will be taken at each iteration when looping over this object. For example:
    ```
    for frameset in Screenshot():
        # `frameset` will contain a single frame with the screenshot here
        time.sleep(1)
    ```
    """

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Frame:
        frame = np.asarray(ImageGrab.grab())
        return Frame.from_array(frame, is_bgr=False)
