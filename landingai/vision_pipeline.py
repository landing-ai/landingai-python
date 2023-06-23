"""The vision pipeline abstraction helps chain image processing operations as sequence of steps. Each step consumes and produces a `FrameSet` which typically contains a source image and derivative metadata and images.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, PrivateAttr

from landingai.common import Prediction
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

_LOGGER = logging.getLogger(__name__)


class Frame(BaseModel):
    """A Frame stores a main image, its metadata and potentially other derived images. This class will be mostly used internally by the FrameSet clase. A pipeline will `FrameSet` since it is more general and a can also keep new `Frames` extracted from existing ones"""

    image: Image.Image
    """Main image generated typically at the beginning of a pipeline"""

    other_images: Dict[str, Image.Image] = {}
    """Other derivative images associated with this the main image. For example: `FrameSet.overlay_predictions` will store the resulting image on `Frame.other_images["overlay"]"""

    predictions: List[Prediction] = []
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

    frames: List[Frame] = []  # Start with empty frame set

    # @classmethod
    # def from_frameset_list(cls, list: List["FrameSet"] = None) -> "FrameSet":
    #     return cls(frames=list)

    @classmethod
    def from_image(
        cls, file: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "FrameSet":
        """Creates a FrameSet from an image file

        Parameters
        ----------
        file : Path to file

        Returns
        -------
        FrameSet : New FrameSet containing a single image
        """

        im = Image.open(file)
        return cls(frames=[Frame(image=im, metadata=metadata)])

    @classmethod
    def from_array(cls, array: np.ndarray, is_BGR: bool = True) -> "FrameSet":
        """Creates a FrameSet from a image encode as ndarray

        Parameters
        ----------
        array : np.ndarray
            Image
        is_BGR : bool, optional
            Assume OpenCV's BGR color space? Defaults to True

        Returns
        -------
        FrameSet
        """
        # img = cv2.cvtColor(np.asarray(self.frames[0].other_images[image_src]), cv2.COLOR_BGR2RGB)
        if is_BGR:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(array)
        return cls(frames=[Frame(image=im)])

    # TODO: Is it worth to emulate a full container? - https://docs.python.org/3/reference/datamodel.html#emulating-container-types
    def __getitem__(self, key: int) -> Frame:
        return self.frames[key]

    def _repr_pretty_(self, pp, cycle) -> str:  # type: ignore
        # Enable a pretty output on Jupiter notebooks `Display()` function
        return str(
            pp.text(
                self.json(
                    # exclude={"frames": {"__all__": {"image", "other_images"}}},
                    indent=2
                )
            )
        )

    def is_empty(self) -> bool:
        """Check if the FrameSet is empty
        Returns
        -------
        bool
            True if the are no Frames on the FrameSet
        """
        return not self.frames  # True if the list is empty

    def run_predict(self, predictor: Predictor) -> "FrameSet":
        """Run a cloud inference model
        Parameters
        ----------
        predictor: the model to be invoked.
        """

        for frame in self.frames:
            frame.predictions = predictor.predict(np.asarray(frame.image))
        return self

    def overlay_predictions(
        self, options: Optional[Dict[str, Any]] = None
    ) -> "FrameSet":  # TODO: Optional where to store
        for frame in self.frames:
            frame.other_images["overlay"] = overlay_predictions(
                frame.predictions, np.asarray(frame.image), options
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

    def show_image(
        self, image_src: str = "", clear_nb_cell: bool = False
    ) -> "FrameSet":
        """Open an a window and display all the images.
        Parameters
        ----------
        image_src: if empty the source image will be displayed. Otherwise the image will be selected from `other_images`
        """
        # TODO: Should show be a end leaf?
        # Check if we are on a notebook context
        try:
            from IPython import display

            for frame in self.frames:
                if clear_nb_cell:
                    display.clear_output(wait=True)
                if image_src == "":
                    display.display(frame.image)
                else:
                    display.display(frame.other_images[image_src])
        except ModuleNotFoundError:
            # Use PIL's implementation
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

    class Config:
        # Add some encoders to prevent large structures from being printed
        json_encoders = {
            np.ndarray: lambda a: f"<np.ndarray: {a.shape}>",
            Image.Image: lambda i: f"<Image.Image: {i.size}>",
        }


# openCV's default VideoCapture cannot drop frames so if the CPU is overloaded the stream will tart to lag behind realtime.
# This class creates a treaded capture implementation that can stay up to date wit the stream and decodes frames only on demand
class NetworkedCamera(BaseModel):
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
    def get_latest_frame(self) -> "FrameSet":
        """Return the most up to date frame by dropping all by the latest frame. This function is blocking"""
        if self.capture_interval is not None:
            t = datetime.now()
            delta = (t - self._last_capture_time).total_seconds()
            if delta <= self.capture_interval:
                time.sleep(delta)
            self._last_capture_time = t
        with self._t_lock:
            ret, frame = self._cap.retrieve()  # non-blocking call
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
