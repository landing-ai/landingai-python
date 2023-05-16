import logging
import time
from datetime import datetime
from pathlib import Path

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

# How many seconds to wait before capturing the next frame
_CAPTURE_FREQUENCY_SECONDS = 3

# TODO: replace below with your own endpoint id and credentials.
api_key = "dvkyqd942h90wn1t3fsbjshsud3xdgs"
api_secret = "gj95e8antnkhcduuwrgok3efrtwpzqojykc05l8yiuxnaecxdqxvawrir0d3yw"
endpoint_id = "ff2aeae5-110f-41df-bbc7-9016f7ec5dcc"

# TODO: replace below url with your own RTSP stream url
stream_url = "rtsp://172.25.101.151/ch0_0.h264"  # This is a Yi Dome Camera


def stream(capture_frame=False, inference_mode=True):
    """Enable camera and start streaming."""
    _LOGGER.info(f"Opening the stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        _LOGGER.error(f"Stream connection is not opened...: {stream_url}")
        cap.release()
        return

    Path(_OUTPUT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    predictor = Predictor(endpoint_id, api_key, api_secret)
    _LOGGER.info(f"Starting the stream: {stream_url}")
    while True:
        time.sleep(_CAPTURE_FREQUENCY_SECONDS)
        # Capture the video frame by frame
        ret, frame = cap.read()
        if not ret:
            _LOGGER.info("Can't receive frame (stream end?). Exiting ...")
            break
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = predictor.predict(frame)
            frame = overlay_predictions(predictions, image=frame)
            filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            frame.save(f"{_OUTPUT_FOLDER_PATH}/infer_{filename}.jpg")
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Display the resulting frame
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _LOGGER.info("Exiting the stream...")
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream(capture_frame=False, inference_mode=True)
