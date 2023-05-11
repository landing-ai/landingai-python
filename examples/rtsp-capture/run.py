import logging
from datetime import datetime
from pathlib import Path
import time

import cv2
from landingai.predict import Predictor
from landingai.visualize import overlay_bboxes

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_LOGGER = logging.getLogger(__name__)

_FRAME_FOLDER_NAME = "captured_frames"
_CAPUTURE_FREQUENCY_SECONDS = 1
_INFERENCE_FREQUENCY_SECONDS = 1

# TODO: polish this example with more documentations

api_key = ""
api_secret = ""
endpoint_id = ""

def stream(capture_frame=False, inference_mode=True):
    """Enable camera and start streaming."""
    # stream_url = "rtsp://172.25.101.151/ch0_0.h264"  # Yi Dome Camera
    stream_url = 0
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Stream connection is not opened...: {stream_url}")
        cap.release()
        return
    fps_count = 0

    Path(_FRAME_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    predictor = Predictor(endpoint_id, api_key, api_secret)
    _LOGGER.info(f"Starting the stream: {stream_url}")
    while True:
        time.sleep(0.5)
        # Capture the video frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if cv2.waitKey(1) & 0xFF == ord("c"):
            capture_frame = not capture_frame
            _LOGGER.info(f"'capture_frame' set to: {capture_frame}")
        if capture_frame and fps_count % _CAPUTURE_FREQUENCY_SECONDS == 0:
            filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            write_succeed = cv2.imwrite(f"captured_frames/frame_{filename}.jpg", frame)
            assert write_succeed, f"Failed to save the image to file: {filename}"
        if inference_mode and fps_count % _INFERENCE_FREQUENCY_SECONDS == 0:
            _LOGGER.info(f"Predicting...")
            dets = predictor.predict(frame)
            frame = overlay_bboxes(dets, image=frame)
            filename = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            write_succeed = cv2.imwrite(f"captured_frames/infer_{filename}.jpg", frame)
            assert write_succeed, f"Failed to save the image to file: {filename}"
        fps_count += 1
        # Display the resulting frame
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _LOGGER.info(f"Exiting the stream...")
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def make_1080p(cap):
    cap.set(3, 1920)
    cap.set(4, 1080)


if __name__ == "__main__":
    stream()
