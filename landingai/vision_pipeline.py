from landingai.visualize import overlay_predictions
from landingai.predict import Predictor
from landingai.common import (
    APICredential,
    ClassificationPrediction,
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)

import numpy as np
from PIL import Image
import cv2

from typing import Dict, List, Tuple, Any, Callable
from pydantic import BaseModel, BaseSettings

class Frame(BaseModel):
    """Single image frame including metadata."""
    image: Image.Image
    """Main image"""

    other_images: Dict[str, Image.Image] = {}
    """Other derivative images associated with this frame (e.g. detection overlay)"""

    predictions: List[Prediction] = []
    """List of predictions for the main image"""

    metadata: Dict[str, Any] = {} 
    """An optional collection of metadata"""

    def run_predict(self, predictor: Predictor) -> "FrameSet":
        """Run a cloud inference model
        Parameters
        ----------
        predictor: the model to be invoked.
        """
        self.predictions=predictor.predict(np.asarray(self.image))
        return self
    
    def to_numpy_array(self) -> "np.ndarray":
        """Return a numpy array on the GRB color format (used by OpenCV)
        """
        return np.asarray(self.image)

    # def __str__(self):
    #     return "member of Test"    
    # def __repr__(self):
    #     return "member of Test"    
    class Config:
        arbitrary_types_allowed = True
    
class FrameSet(BaseModel):
    """Sequence of frames and metadata."""
    frames:List[Frame] = [] # Start with empty frame set

    @classmethod
    def fromImage(cls, file: str) -> "FrameSet":
        im = Image.open(file)
        # im = cv2.imread(file)
        # if im is None:
        #     raise Exception(f"Could not open image ({file})")
        return cls(frames=[Frame(image=im)])

    @classmethod
    def fromArray(cls, array: np.ndarray, is_BGR:bool = True) -> "FrameSet":
        # img = cv2.cvtColor(np.asarray(self.frames[0].other_images[image_src]), cv2.COLOR_BGR2RGB)
        if is_BGR:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(array)
        return cls(frames=[Frame(image=im)])

    # TODO: Is it worth to emulate a full container? - https://docs.python.org/3/reference/datamodel.html#emulating-container-types
    def __getitem__(self, key):
        return self.frames[key]

    def run_predict(self, predictor: Predictor) -> "FrameSet":
        """Run a cloud inference model
        Parameters
        ----------
        predictor: the model to be invoked.
        """

        for frame in self.frames:
            frame.predictions=predictor.predict(np.asarray(frame.image))
        return self

    def overlay_predictions(self) -> "FrameSet":  # TODO: Optional where to store
        for frame in self.frames:
            frame.other_images["overlay"]=overlay_predictions(frame.predictions, np.asarray(frame.image))
        return self
    
    def resize(self, width:int = None, height: int = None) -> "FrameSet":  # TODO: Optional where to store
        """Returns a resized copy of this image. If width or height is missing the resize will preserve the aspect ratio
        Parameters
        ----------
        width: The requested width in pixels.
        height: The requested width in pixels.
        """
        for frame in self.frames:
            # Compute the final dimensions on the first image
            if width is None: 
                width  = int(height * float(frame.image.size[0]/frame.image.size[1]) )
            if height is None:
                height = int(width * float(frame.image.size[1]/frame.image.size[0]) )
            frame.image=frame.image.resize((width, height))
        return self

    def downsize(self, width:int = None, height: int = None) -> "FrameSet":  # TODO: Optional where to store
        """Resize only if the image is larger than the expected dimensions,
        Parameters
        ----------
        width: The requested width in pixels.
        height: The requested width in pixels.
        """
        for frame in self.frames:
            # Compute the final dimensions on the first image
            if width is None: 
                width  = int(height * float(frame.image.size[0]/frame.image.size[1]) )
            if height is None:
                height = int(width * float(frame.image.size[1]/frame.image.size[0]) )
            if frame.image.size[0]>width or frame.image.size[1]>height:
                frame.image=frame.image.resize((width, height))
        return self

    def show_image(self, image_src: str = "") -> "FrameSet":
        """Open an OpenCV window and display all the images. This call will stop the execution until a key is pressed.
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
    
    def apply(self, function:Callable[[Frame], Frame] = lambda f: f) -> "FrameSet":
        """Apply function to all frames"""
        for i in range(len(self.frames)):
            self.frames[i]=function(self.frames[i])
        return self
    def filter(self, function:Callable[[Frame], bool] = lambda f: True) -> "FrameSet":
        """Filter predictions"""
        for i in reversed(range(0,len(self.frames))): # Traverse in reverse so we can delete
            if not function(self.frames[i]):
                self.frames.pop(i)
        return self

    
import threading
import time

import cv2

# openCV's default VideoCapture cannot drop frames so if the CPU is overloaded the stream will tart to lag behind realtime.
# This class creates a treaded capture implementation that can stay up to date wit the stream and decodes frames only on demand
class NetworkedCamera:
    stream_url: str
    enable_motion_detection: bool
    motion_detection_threshold: int
    forced_framerate: int
    previous_frame: Frame = None

    def __init__(self, stream_url:str
                 , enable_motion_detection:bool = False, motion_detection_threshold:int = 2
                 , forced_framerate:int = None):
        self.stream_url = stream_url
        self.enable_motion_detection = enable_motion_detection
        self.motion_detection_threshold = motion_detection_threshold
        self.forced_framerate = forced_framerate
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            self.cap.release()
            raise Exception(f"Could not open stream ({stream_url})")
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / self.cap.get(cv2.CAP_PROP_FPS)
        self.FPS_MS = int(self.FPS * 1000)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffering to 1 frames
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def __del__(self):
        self.cap.release()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
                time.sleep(self.FPS)  # Limit acquisition speed
            if not ret:
                raise Exception(f"Connection to camera broken ({self.stream_url})")


    # retrieve latest frame
    def get_latest_frame(self):
        """Return the most up to date frame by dropping all by the latest frame. This function is blocking
        """        
        with self.lock:
            ret, frame = self.cap.retrieve()
            if not ret:
                raise Exception(f"Connection to camera broken ({self.stream_url})")
            if self.enable_motion_detection:
                if self._detect_motion(frame):
                    return FrameSet.fromArray(frame)
                else:
                    return FrameSet() # Empty frame
                                
        return FrameSet.fromArray(frame)
    
    previous_frame: Frame

    def _detect_motion(self, frame) -> bool:
        """
        """        
        # Prepare image; grayscale and blur
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(
            src=prepared_frame, ksize=(5, 5), sigmaX=0
        )
        
        if self.previous_frame is None:
            # Save the result for the next invocation
            self.previous_frame = prepared_frame
            return True # First frame; there is no previous one yet

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
            previous_frame = prepared_frame
            return True
        return False

    # Make the class iterable
    def __iter__(self):
        return self    

    def __next__(self):
        return self.get_latest_frame()
