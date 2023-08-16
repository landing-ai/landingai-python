import logging
from datetime import datetime
import time
import functools

from landingai.predict import Predictor, EdgePredictor
from landingai.pipeline.image_source import NetworkedCamera
from landingai.predict import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_LOGGER = logging.getLogger(__name__)

# How many frames per second should we capture and process
_CAPTURE_INTERVAL = (
    0.1  # In milliseconds. Set to 1 if you want to capture at the max prediction rate XXX seconds
)

# Public Cloud & Sky detection segmentation model
api_key = "land_sk_aMemWbpd41yXnQ0tXvZMh59ISgRuKNRKjJEIUHnkiH32NBJAwf"
endpoint_id = "432d58f6-6cd4-4108-a01c-2f023503d838"

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
stream_url = (
    "https://itsstreamingbr.dotd.la.gov/public/br-cam-110.streams/playlist.m3u8"
)


if __name__ == "__main__":
    # Cloud inference model to segment clouds
    cloud_sky_model = Predictor(endpoint_id, api_key=api_key)
    # Local inference model example. In order to use it, you need to manually run the local inference server with the "cloud & sky" model.
    # cloud_sky_model = EdgePredictor()

    Camera = NetworkedCamera(
        stream_url, motion_detection_threshold=1, capture_interval=_CAPTURE_INTERVAL
    )
    _LOGGER.info("Starting")
    start_time = datetime.now()
    for frame in Camera:
        _LOGGER.info(
            f"Aquisition time {(datetime.now()-start_time).total_seconds():.5f} sec"
        )
        frame = frame.downsize(width=1024)
        start_time = datetime.now()
        frame = frame.run_predict(predictor=cloud_sky_model)
        _LOGGER.info(
            f"Inference time {(datetime.now()-start_time).total_seconds():.2f} sec"
        )
        start_time = datetime.now()

        # .overlay_predictions()
        # .show_image()
        # .show_image(image_src="overlay")
        # .save_image(filename_prefix="./capture")
