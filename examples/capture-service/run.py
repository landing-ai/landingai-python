import logging
import time
from datetime import datetime
from pathlib import Path
from capture import ThreadedCapture

import cv2
import numpy as np

from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_LOGGER = logging.getLogger(__name__)

# Captured frames will be saved in this folder
_OUTPUT_FOLDER_PATH = "examples/output/capture-service"

# How many frames per second should we capture and process
_CAPTURE_FPS = 2

# TODO: replace below with your own endpoint id and credentials.
api_key = "dvkyqd942h90wn1t3fsbjshsud3xdgs"
api_secret = "gj95e8antnkhcduuwrgok3efrtwpzqojykc05l8yiuxnaecxdqxvawrir0d3yw"
endpoint_id = "ff2aeae5-110f-41df-bbc7-9016f7ec5dcc"

# TODO: replace below url with your own RTSP stream url
# Apple test stream
# stream_url = "http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8"  # This is a Yi Dome Camera
# Dexter Avenue AL - https://www.wsfa.com/weather/cams/
stream_url = "https://s78.ipcamlive.com/streams/4etgocfj23fhnhsne/stream.m3u8" 


def stream(capture_frame=False, inference_mode=True):
    """Enable camera and start streaming."""
    _LOGGER.info(f"Opening the stream: {stream_url}")
    threaded_camera = ThreadedCapture(stream_url,_CAPTURE_FPS)
    Path(_OUTPUT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    predictor = Predictor(endpoint_id, api_key, api_secret)
    while True:
        # Use threaded_camera to skip frames and get the latest
        ret, frame = threaded_camera.read()
        if not ret:
            _LOGGER.info("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Resize image if needed to Keep width t0 1024
        if frame.shape[1]>1024: 
            width = 1024 # Set constant width
            height = int(frame.shape[0] * (1024/frame.shape[1]) ) # Keep aspect ratio
            dim = (width, height)    
            # resize image
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
 
        # Press c to capture a frame
        if cv2.waitKey(1) & 0xFF == ord("c"):
            capture_frame = not capture_frame
            _LOGGER.info(f"'capture_frame' set to: {capture_frame}")
        if capture_frame:
            filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            write_succeed = cv2.imwrite(
                f"{_OUTPUT_FOLDER_PATH}/frame_{filename}.jpg", frame
            )
            assert write_succeed, f"Failed to save the image to file: {filename}"
        if inference_mode:
            _LOGGER.info("Predicting...")
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = predictor.predict(frame)
            frame = overlay_predictions(predictions, image=frame)
            # filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            # frame.save(f"{_OUTPUT_FOLDER_PATH}/infer_{filename}.jpg")
            frame = np.array(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Display the resulting frame
        cv2.imshow("window", frame)
        # Press q to quit
        if cv2.waitKey(1 if inference_mode else threaded_camera.FPS_MS) & 0xFF == ord("q"):
            _LOGGER.info("Exiting the stream...")
            break
# XXX Todo: waitkey are to fast for c & q.... why is the color changed... 

    # When everything done, release the capture
    del threaded_camera
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream(capture_frame=False, inference_mode=True)
