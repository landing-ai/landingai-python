import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from requests import Session
from requests.adapters import HTTPAdapter, Retry

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """A class that calls your inference endpoint on the Landing AI platform."""

    _url: str = "https://predict.app.landing.ai/inference/v1/predict"

    def __init__(self, endpoint_id: str, api_key: Optional[str], api_secret: Optional[str]) -> None:
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
        session.mount(api_url, HTTPAdapter(max_retries=retries))
        session.headers.update(
            {
                "apikey": self._api_key,
                "apisecret": self._api_secret,
                "contentType": "application/json",
            }
        )
        return session

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Call the inference endpoint and return the prediction result.

        Parameters
        ----------
        image: the input image to be predicted

        Returns
        -------
        The inference result in a list of dictionary. Each dictionary is a prediction result.
        The ininference result has been filtered by the confidence threshold
        set in the Landing AI platform and sorted by confidence score in descending order.
        """
        img = cv2.imencode(".png", image)[1]
        files = [("file", ("image.png", img, "image/png"))]
        s = self._create_session(self._url)
        payload = {"endpoint_id": self._endpoint_id}
        headers = {
                "apikey": self._api_key,
                "apisecret": self._api_secret,
                "contentType": "application/json",
            }
        response = s.post(Predictor._url, headers=headers, files=files, params=payload)
        response.raise_for_status()
        json_dict = response.json()
        _LOGGER.debug("Response: %s", json_dict)
        return _extract_prediction(json_dict)


def _extract_prediction(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    response_type = response["backbonetype"]
    predictions = PREDICTION_EXTRACTOR[response_type](response)

    return predictions


def _extract_od_prediction(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract Object Detection result from response

    Parameters
    ----------
    response: response from the prediction API
    Example example input:
    {
        "backbonetype": "ObjectDetectionPrediction",
        "backbonepredictions":
        {
            "196f6705-4486-42e4-8b48-325bba622f18":
            {
                "score": 0.9939298629760742,
                "defect_id": 72558,
                "coordinates":
                {
                    "xmin": 1523,
                    "ymin": 1417,
                    "xmax": 1982,
                    "ymax": 1801
                },
                "labelIndex": 1,
                "labelName": "Screw"
            },
            "ca576098-5d97-4d6e-87e5-e3c93e0187d3":
            {
                "score": 0.9845893383026123,
                "defect_id": 72558,
                "coordinates":
                {
                    "xmin": 429,
                    "ymin": 1035,
                    "xmax": 651,
                    "ymax": 1207
                },
                "labelIndex": 1,
                "labelName": "Screw"
            },
            "cdb4941b-db69-4952-8769-e057a470a94a":
            {
                "score": 0.9666865468025208,
                "defect_id": 72558,
                "coordinates":
                {
                    "xmin": 948,
                    "ymin": 1595,
                    "xmax": 1123,
                    "ymax": 1804
                },
                "labelIndex": 1,
                "labelName": "Screw"
            }
        },
        "predictions":
        {
            "score": 0.9939298629760742,
            "labelIndex": 1,
            "labelName": "NG"
        },
        "type": "ClassificationPrediction",
        "latency":
        {
            "preprocess_s": 0.005197763442993164,
            "infer_s": 0.16716742515563965,
            "postprocess_s": 0.0006124973297119141,
            "serialize_s": 0.015230655670166016,
            "input_conversion_s": 0.05388998985290527,
            "model_loading_s": 0.0002772808074951172
        },
        "model_id": "1df69515-2218-4fdb-b1be-13694cb8e162"
    }    

    Returns
    -------
    A list of Object Detection predictions, each prediction is a dictionary with the following keys:
        id: str
        label: str
        confidence_score: float
        bboxes: Dict[str, int]
        E.g. {
            "id": "cdb4941b-db69-4952-8769-e057a470a94a",
            "label": "Screw",
            "confidence_score": 0.9666865468025208,
            "bboxes": {
                "xmin": 948,
                "ymin": 1595,
                "xmax": 1123,
                "ymax": 1804,
            }
        }
    """
    predictions = response["backbonepredictions"]
    return [
        {
            "id": id,
            "label": prediction["labelName"],
            "confidence_score": prediction["score"],
            "bboxes": prediction["coordinates"],
        }
        for id, prediction in predictions.items()
    ]


PREDICTION_EXTRACTOR = {
    "ObjectDetectionPrediction": _extract_od_prediction,
}
