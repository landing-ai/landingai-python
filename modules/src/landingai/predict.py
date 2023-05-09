import logging
from typing import Any, Dict, List

import cv2
import numpy as np
from requests import Session
from requests.adapters import HTTPAdapter, Retry

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """A class that calls the Landing AI prediction API"""

    _url: str = "https://predict.app.landing.ai/inference/v1/predict"

    def __init__(self, endpoint_id: str, api_key: str, api_secret: str) -> None:
        self._endpoint_id = endpoint_id
        self._api_key = api_key
        self._api_secret = api_secret

    def _create_session(self, api_url: str) -> Session:
        """Create a requests session with retry"""
        session = Session()
        retries = Retry(
            # TODO: make them configurable
            total=5,
            backoff_factor=3,
            status_forcelist=[408, 413, 429, 500, 502, 503, 504],
        )
        session.mount(self._url, HTTPAdapter(max_retries=retries))
        session.headers.update(
            {
                "apikey": self._api_key,
                "apisecret": self._api_secret,
                "contentType": "application/json",
            }
        )
        return session

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # TODO: change to PNG
        img = cv2.imencode(".jpg", image)[1]
        files = [("file", ("image.jpg", img, "image/jpg"))]
        s = self._create_session(self._url)
        payload = {"endpoint_id": self._endpoint_id}
        response = s.post(Predictor._url, files=files, params=payload)
        try:
            response.raise_for_status()
        except Exception as e:
            _LOGGER.error(
                f"Failed calling predict endpoint ({self._url}) due to error: {e}"
            )
            return []
        json_dict = response.json()
        return _extract_prediction(json_dict)


def _extract_prediction(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    response_type = response["type"]
    predictions = PREDICTION_EXTRACTOR[response_type](response)

    return predictions


def _extract_od_prediction(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract Object Detection result from response

    Parameters
    ----------
    response: response from the prediction API
    Example example input:
    {
        "backbonetype": "None",
        "backbonepredictions": "None",
        "predictions":
        {
            "e2a909fb-5d65-44c9-8cf1-09f999a3bc96":
            {
                "score": 0.9830147624015808,
                "defect_id": 2525478,
                "coordinates":
                {
                    "xmin": 1332,
                    "ymin": 200,
                    "xmax": 1916,
                    "ymax": 784
                },
                "labelIndex": 1,
                "labelName": "Garage open"
            }
        },
        "type": "ObjectDetectionPrediction",
        "latency":
        {
            "preprocess_s": 0.004376649856567383,
            "infer_s": 0.1019594669342041,
            "postprocess_s": 0.000013589859008789062,
            "serialize_s": 0.003810882568359375,
            "input_conversion_s": 0.03130197525024414,
            "model_loading_s": 0.00025391578674316406
        },
        "model_id": "60c19e47-36ee-4309-91c0-1bcce137edd7"
    }

    Returns
    -------
    A list of Object Detection predictions, each prediction is a dictionary with the following keys:
        id: str
        label: str
        confidence_score: float
        bounding_box: Dict[str, int]
    """
    predictions = response["predictions"]
    return [
        {
            "id": id,
            "label": prediction["labelName"],
            "confidence_score": prediction["score"],
            "bounding_box": prediction["coordinates"],
        }
        for id, prediction in predictions.items()
    ]


PREDICTION_EXTRACTOR = {
    "ObjectDetectionPrediction": _extract_od_prediction,
}
