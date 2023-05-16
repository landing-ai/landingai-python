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
_CAPTURE_INTERVAL = 1  # In milliseconds. Set to 1 if you want to capture at the max prediction rate

# Public Cloud & Sky detection segmentation model
api_key         = "bu8y8czyonaip6ceov75nfnlpnr9blh"  
api_secret      = "mdebq6hxq19fg86k3p53rwcxh16h2qudcfonl6sjrde334y2vxz4qj4wnefh05"  
endpoint_id     = "432d58f6-6cd4-4108-a01c-2f023503d838"  

#
# Below we provide some links to public cameras. Local RTSP cameras can also be used by specifying a local URL 
# like "rtsp://172.25.101.151/ch0_0.h264". In order to find the URL for your camera, this page is a good start 
# https://www.ispyconnect.com/cameras

# Apple test stream
# stream_url = "http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8"  
# Dexter Avenue AL - https://www.wsfa.com/weather/cams/
# stream_url = "https://s78.ipcamlive.com/streams/4etrywblgzguidm6v/stream.m3u8" 
# Louisiana Department of Transportation - https://511la.org/
# stream_url = "https://itsstreamingbr.dotd.la.gov/public/br-cam-015.streams/playlist.m3u8" 
stream_url = "https://itsstreamingbr.dotd.la.gov/public/br-cam-110.streams/playlist.m3u8" 


def stream(capture_frame=False, inference_mode=True):
    """Enable camera and start streaming."""
    _LOGGER.info(f"Opening the stream: {stream_url}")
    threaded_camera = ThreadedCapture(stream_url)
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
 
        if capture_frame:
            filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            write_succeed = cv2.imwrite(
                f"{_OUTPUT_FOLDER_PATH}/frame_{filename}.jpg", frame
            )
            assert write_succeed, f"Failed to save the image to file: {filename}"

        if inference_mode:
            _LOGGER.info("Predicting...")
            start = time.time()
            predictions = predictor.predict(frame)
            frame = overlay_predictions(predictions, image=frame) 
            frame = np.array(frame)
            _LOGGER.info(f"Prediction completed in {time.time() - start} seconds")
        # Display the resulting frame
        cv2.imshow("window", frame)

        key = cv2.waitKey(_CAPTURE_INTERVAL) & 0xFF
        # Press c to capture a frame (this is a toggle)
        if key == ord("c"):
            capture_frame = not capture_frame
            _LOGGER.info(f"'capture_frame' set to: {capture_frame}")
        # Press q to quit
        if key == ord("q"):
            _LOGGER.info("Exiting the stream...")
            break

    # When everything done, release the capture
    del threaded_camera
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream(capture_frame=True, inference_mode=True)
