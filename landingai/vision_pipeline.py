"""The vision pipeline is a layer that allows chaining image processing operations as a sequential pipeline. The main class passed throughout a pipeline is the `FrameSet` which typically contains a source image and derivative metadata and images.
"""

from landingai.visualize import overlay_predictions
from landingai.predict import Predictor
from landingai.common import Prediction

import numpy as np
from PIL import Image
import cv2

from datetime import datetime
import logging
import time
import threading
from typing import Dict, Any, Callable, Union
from pydantic import BaseModel, PrivateAttr

_LOGGER = logging.getLogger(__name__)


class Frame(BaseModel):
    """A Frame stores a main image, its metadata and potentially derived images. This class will be mostly used internally by the FrameSet."""

    image: Image.Image
    """Main image"""

    other_images: Dict[str, Image.Image] = {}
    """Other derivative images associated with this frame (e.g. detection overlay)"""

    predictions: list[Prediction] = []
    """List of predictions for the main image"""

    metadata: Dict[str, Any] = {}
    """An optional collection of metadata"""

    def run_predict(self, predictor: Predictor) -> "Frame":
        """Run a cloud inference model
        Parameters
        ----------
        predictor: the model to be invoked.
        """
        self.predictions = predictor.predict(np.asarray(self.image))
        return self

    def to_numpy_array(self) -> np.ndarray:
        """Return a numpy array using GRB color encoding (used by OpenCV)"""
        return np.asarray(self.image)

    class Config:
        arbitrary_types_allowed = True


class FrameSet(BaseModel):
    """A FrameSet is a collection of frames (in order). Typically a FrameSet will include a single image but there are circumstances where other images will be extracted from the initial one. For example: we may want to identify vehicles on an initial image and then extract sub-images for each of the vehicles."""

    frames: list[Frame] = []  # Start with empty frame set

    @classmethod
    def from_image(cls, file: str) -> "FrameSet":
        im = Image.open(file)
        return cls(frames=[Frame(image=im)])

    @classmethod
    def from_array(cls, array: np.ndarray, is_BGR: bool = True) -> "FrameSet":
        # img = cv2.cvtColor(np.asarray(self.frames[0].other_images[image_src]), cv2.COLOR_BGR2RGB)
        if is_BGR:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(array)
        return cls(frames=[Frame(image=im)])

    # TODO: Is it worth to emulate a full container? - https://docs.python.org/3/reference/datamodel.html#emulating-container-types
    def __getitem__(self, key: int) -> Frame:
        return self.frames[key]

    def run_predict(self, predictor: Predictor) -> "FrameSet":
        """Run a cloud inference model
        Parameters
        ----------
        predictor: the model to be invoked.
        """

        for frame in self.frames:
            frame.predictions = predictor.predict(np.asarray(frame.image))
        return self

    def overlay_predictions(self) -> "FrameSet":  # TODO: Optional where to store
        for frame in self.frames:
            frame.other_images["overlay"] = overlay_predictions(
                frame.predictions, np.asarray(frame.image)
            )
        return self

    def resize(
        self, width: Union[int, None] = None, height: Union[int, None] = None
    ) -> "FrameSet":  # TODO: Optional where to store
        """Returns a resized copy of this image. If width or height is missing the resize will preserve the aspect ratio
        Parameters
        ----------
        width: The requested width in pixels.
        height: The requested width in pixels.
        """
        if width is None and height is None:  # No resize needed
            return self
        for frame in self.frames:
            # Compute the final dimensions on the first image
            if width is None:
                width = int(height * float(frame.image.size[0] / frame.image.size[1]))  # type: ignore
            if height is None:
                height = int(width * float(frame.image.size[1] / frame.image.size[0]))
            frame.image = frame.image.resize((width, height))
        return self

    def downsize(
        self, width: Union[int, None] = None, height: Union[int, None] = None
    ) -> "FrameSet":  # TODO: Optional where to store
        """Resize only if the image is larger than the expected dimensions,
        Parameters
        ----------
        width: The requested width in pixels.
        height: The requested width in pixels.
        """
        if width is None and height is None:  # No resize needed
            return self
        for frame in self.frames:
            # Compute the final dimensions on the first image
            if width is None:
                width = int(height * float(frame.image.size[0] / frame.image.size[1]))  # type: ignore
            if height is None:
                height = int(width * float(frame.image.size[1] / frame.image.size[0]))
            if frame.image.size[0] > width or frame.image.size[1] > height:
                frame.image = frame.image.resize((width, height))
        return self

    def save_image(self, filename_prefix: str, image_src: str = "") -> "FrameSet":
        """Save all the images on the FrameSet to disk (as PNG)

        Parameters
        ----------
        filename_prefix : path and name prefix for the image file
        image_src : if empty the source image will be displayed. Otherwise the image will be selected from `other_images`
        """
        timestamp = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )  # TODO saving faster than 1 sec will cause image overwrite
        c = 0
        for frame in self.frames:
            img = frame.image if image_src == "" else frame.other_images[image_src]
            img.save(f"{filename_prefix}_{timestamp}_{image_src}_{c}.png", format="PNG")
            c += 1
        return self

    def show_image(self, image_src: str = "") -> "FrameSet":
        """Open an a window and display all the images.
        Parameters
        ----------
        image_src: if empty the source image will be displayed. Otherwise the image will be selected from `other_images`
        """
        # TODO: Should show be a end leaf?
        for frame in self.frames:
            if image_src == "":
                frame.image.show()
            else:
                frame.other_images[image_src].show()

        # # TODO: Implement image stacking when we have multiple frames (https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/)
        # """Open an OpenCV window and display all the images. This call will stop the execution until a key is pressed.
        # Parameters
        # ----------
        # image_src: if empty the source image will be displayed. Otherwise the image will be selected from `other_images`
        # """
        # # OpenCV is full of issues when it comes to displaying windows (see https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
        # cv2.namedWindow("image")
        # cv2.startWindowThread()
        # if image_src == "":
        #     img = cv2.cvtColor(np.asarray(self.frames[0].image), cv2.COLOR_BGR2RGB)
        # else:
        #     img = cv2.cvtColor(np.asarray(self.frames[0].other_images[image_src]), cv2.COLOR_BGR2RGB)
        # cv2.imshow("Landing AI - Press any key to exit", img)
        # cv2.waitKey(0) # close window when a key press is detected
        # cv2.waitKey(1)
        # cv2.destroyWindow('image')
        # for i in range (1,5):
        #     cv2.waitKey(1)

        return self

    def apply(self, function: Callable[[Frame], Frame] = lambda f: f) -> "FrameSet":
        """Apply a function to all frames

        Parameters
        ----------
        function: lambda function that takes individual frames and returned an updated frame
        """
        for i in range(len(self.frames)):
            self.frames[i] = function(self.frames[i])
        return self

    def filter(self, function: Callable[[Frame], bool] = lambda f: True) -> "FrameSet":
        """Evaluate a function on every frame and keep or remove

        Parameters
        ----------
        function : lambda function that gets invoked on every Frame. If it returns False, the Frame will be deleted
        """
        for i in reversed(
            range(0, len(self.frames))
        ):  # Traverse in reverse so we can delete
            if not function(self.frames[i]):
                self.frames.pop(i)
        return self


# openCV's default VideoCapture cannot drop frames so if the CPU is overloaded the stream will tart to lag behind realtime.
# This class creates a treaded capture implementation that can stay up to date wit the stream and decodes frames only on demand
class NetworkedCamera(BaseModel):
    """The NetworkCamera class can connect to RTSP and other live video sources in order to grab frames. The main concern is to be able to consume frames at the source speed and drop them as needed to ensure the application allday gets the lastes frame"""

    stream_url: str
    motion_detection_threshold: int
    capture_interval: float
    previous_frame: Union[Frame, None] = None
    _last_capture_time: datetime = PrivateAttr()
    _cap: Any = PrivateAttr()  # cv2.VideoCapture
    _FPS: int = PrivateAttr()
    _lock: Any = PrivateAttr()  # threading.Lock
    _t: Any = PrivateAttr()  # threading.Thread

    def __init__(
        self,
        stream_url: str,
        motion_detection_threshold: int = 0,
        capture_interval: Union[float, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        stream_url : url to video source
        motion_detection_threshold : If set to zero then motion detections is disabled. Any other value (0-100) will make the camera drop all images that don't have significant changes
        capture_interval : If set to None, the NetworkedCamera will acquire images as fast as the source permits. Otherwise will grab a new frame every capture_interval seconds
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            cap.release()
            raise Exception(f"Could not open stream ({stream_url})")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffering to 1 frames

        super().__init__(
            stream_url=stream_url,
            motion_detection_threshold=motion_detection_threshold,
            capture_interval=capture_interval,
        )
        self._last_capture_time = datetime.now()
        # FPS = 1/X
        # self.FPS_MS = int(self.FPS * 1000)
        self._FPS = 1 / cap.get(cv2.CAP_PROP_FPS)  # Get the source's framerate
        self._cap = cap
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._reader)
        self._t.daemon = True
        self._t.start()

    def __del__(self) -> None:
        self._cap.release()

    # grab frames as soon as they are available
    def _reader(self) -> None:
        while True:
            with self._lock:
                ret = self._cap.grab()
                time.sleep(self._FPS)  # Limit acquisition speed
            if not ret:
                raise Exception(f"Connection to camera broken ({self.stream_url})")

    # retrieve latest frame
    def get_latest_frame(self) -> "FrameSet":
        """Return the most up to date frame by dropping all by the latest frame. This function is blocking"""
        with self._lock:
            if self.capture_interval is not None:
                t = datetime.now()
                delta = (t - self._last_capture_time).total_seconds()
                if delta <= self.capture_interval:
                    time.sleep(delta)
                self._last_capture_time = t

            ret, frame = self._cap.retrieve()
            if not ret:
                raise Exception(f"Connection to camera broken ({self.stream_url})")
            if self.motion_detection_threshold > 0:
                if self._detect_motion(frame):
                    return FrameSet.from_array(frame)
                else:
                    return FrameSet()  # Empty frame

        return FrameSet.from_array(frame)

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

    def __next__(self) -> "FrameSet":
        return self.get_latest_frame()

    class Config:
        arbitrary_types_allowed = True
