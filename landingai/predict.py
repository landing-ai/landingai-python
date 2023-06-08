import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import requests
from pydantic import ValidationError
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from landingai.common import (
    APICredential,
    ClassificationPrediction,
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)
from landingai.exceptions import HttpResponse

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """A class that calls your inference endpoint on the LandingLens platform."""

    _url: str = "https://predict.app.landing.ai/inference/v1/predict"
    _num_retry: int = 3

    def __init__(
        self,
        endpoint_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """Predictor constructor

        Parameters
        ----------
        endpoint_id
            A unique string that identifies your inference endpoint.
            This string can be found in the URL of your inference endpoint.
            Example: "9f237028-e630-4576-8826-f35ab9000abc" is the endpoint id in this URL:
            https://predict.app.landing.ai/inference/v1/predict?endpoint_id=9f237028-e630-4576-8826-f35ab9000abc
        api_key
            The API Key of your LandingLens organization.
            If not provided, it will try to load from the environment variable
            LANDINGAI_API_KEY or from the .env file.
        api_secret
            The API Secret of your LandingLens organization.
            If not provided, it will try to load from the environment variable
            LANDINGAI_API_SECRET or from the .env file.
        """
        self._endpoint_id = endpoint_id
        self._api_credential = self._load_api_credential(api_key, api_secret)
        self._session = self._create_session()

    def _load_api_credential(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> APICredential:
        """Create an APICredential object from the provided api_key and api_secret if they are not none.
        Otherwise, it will try to load from the environment variables or .env file.
        """
        if api_key is None or api_secret is None:
            try:
                api_credential = APICredential()
            except ValidationError:
                raise ValueError("API credential is not provided")
        else:
            api_credential = APICredential(api_key=api_key, api_secret=api_secret)
        return api_credential

    def _create_session(self) -> Session:
        """Create a requests session with retry"""
        session = Session()
        retries = Retry(
            # TODO: make them configurable
            total=self._num_retry,
            backoff_factor=3,
            raise_on_redirect=True,
            raise_on_status=False,  # We are already raising exceptions during backend invocations
            allowed_methods=["GET", "POST", "PUT"],
            status_forcelist=[
                # 408 Request Timeout , 413 Content Too Large, 500 Internal Server Error
                429,  # Too Many Requests  (ie. rate limiter)
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ],
        )
        session.mount(
            Predictor._url, HTTPAdapter(max_retries=retries)
        )  # Since POST is not idempotent we will ony retry on the this specific API
        session.headers.update(
            {
                "apikey": self._api_credential.api_key,
                "apisecret": self._api_credential.api_secret,
                "contentType": "application/json",
            }
        )
        return session

    def predict(
        self, image: Union[np.ndarray, PIL.Image.Image], **kwargs: Any
    ) -> List[Prediction]:
        """Call the inference endpoint and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted. The image should be in the RGB format if it has three channels.

        Returns
        -------
        The inference result in a list of dictionary
            Each dictionary is a prediction result.
            The inference result has been filtered by the confidence threshold set in LandingLens and sorted by confidence score in descending order.
        """
        buffer_bytes = self._serialize_image(image)
        files = [("file", ("image.png", buffer_bytes, "image/png"))]
        payload = {"endpoint_id": self._endpoint_id, "device_type": "pylib"}
        return self._do_post(Predictor._url, files, payload)

    def _do_post(
        self, endpoint_url: str, files: List[Any], payload: Dict[str, Any]
    ) -> List[Prediction]:
        """Call the inference endpoint and extract the prediction result."""
        try:
            resp = self._session.post(endpoint_url, files=files, params=payload)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to the model server. Please double check the model server url ({Predictor._url}) is correct.\nException detail: {e}"
            )
        response = HttpResponse.from_response(resp)
        _LOGGER.debug("Response: %s", response)
        response.raise_for_status()
        json_dict = response.json()
        return _extract_prediction(json_dict)

    def _serialize_image(self, image: Union[np.ndarray, PIL.Image.Image]) -> bytes:
        if image is None or (isinstance(image, np.ndarray) and len(image) == 0):
            raise ValueError(f"Input image must be non-emtpy, but got: {image}")
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        buffer_bytes = img_buffer.getvalue()
        img_buffer.close()
        return buffer_bytes


class OcrPredictor(Predictor):
    _url: str = "https://predict.app.landing.ai/inference/v1/detect-text"

    def __init__(
        self,
        # TODO: remove det_mode
        det_mode: str,
        threshold: float,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self._det_mode = det_mode
        self._threshold = threshold
        self._api_crendential = self._load_api_credential(api_key, api_secret)
        self._session = self._create_session()

    def predict(
        self, image: Union[np.ndarray, PIL.Image.Image], **kwargs: Any
    ) -> List[Prediction]:
        """Run OCR on the input image and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted
        regions_of_interest
            A list of region of interest boxes/quadrilateral. Each box is a list of 4 points (x, y).
            If it is None, the whole image will be used as the region of interest.

        Returns
        -------
        List[OcrPrediction]
            A list of OCR prediction result.
        """

        # TODO: crop image to ROI
        buffer_bytes = self._serialize_image(image)
        files = [("files", ("image.png", buffer_bytes, "image/png"))]
        payload: Dict[str, Any] = {"device_type": "pylib"}
        if "regions_of_interest" in kwargs:
            rois: List[List[Tuple[int, int]]] = kwargs["regions_of_interest"]
            payload["regions_of_interest"] = [rois]
        return self._do_post(OcrPredictor._url, files, payload)


def _extract_prediction(response: Dict[str, Any]) -> List[Prediction]:
    response_type = response["backbonetype"]
    if response_type is None and response["type"] == "SegmentationPrediction":
        response_type = "SegmentationPredictionVP"  # Visual Prompting response
    if response_type is None:
        response_type = response["type"]  # Classification response
    predictions = PREDICTION_EXTRACTOR[response_type](response)
    return predictions  # type: ignore


def _extract_class_prediction(
    response: Dict[str, Any]
) -> List[ClassificationPrediction]:
    """Extract Classification prediction result from response

    Parameters
    ----------
    response: Response from the LandingLens prediction endpoint.
    Example input:
    {
        "backbonetype": None,
        "backbonepredictions": None,
        "predictions":
        {
            "score": 0.9951885938644409,
            "labelIndex": 0,
            "labelName": "Fire"
        },
        "type": "ClassificationPrediction",
        "latency":
        {
            "preprocess_s": 0.0035605430603027344,
            "infer_s": 0.11771035194396973,
            "postprocess_s": 0.0000457763671875,
            "serialize_s": 0.00015735626220703125,
            "input_conversion_s": 0.002260446548461914,
            "model_loading_s": 5.1028594970703125
        },
        "model_id": "8a7cbf5d-a869-4c85-b4c8-b4f9bd937117"
    }
    """
    return [
        ClassificationPrediction(
            score=response["predictions"]["score"],
            label_index=response["predictions"]["labelIndex"],
            label_name=response["predictions"]["labelName"],
        )
    ]


def _extract_od_prediction(response: Dict[str, Any]) -> List[ObjectDetectionPrediction]:
    """Extract Object Detection prediction result from response

    Parameters
    ----------
    response: Response from the LandingLens prediction endpoint.
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
        score: float
        bboxes: Dict[str, int]
        E.g. {
            "id": "cdb4941b-db69-4952-8769-e057a470a94a",
            "label": "Screw",
            "score": 0.9666865468025208,
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
        ObjectDetectionPrediction(
            id=id,
            label_name=pred["labelName"],
            label_index=pred["labelIndex"],
            score=pred["score"],
            bboxes=(
                pred["coordinates"]["xmin"],
                pred["coordinates"]["ymin"],
                pred["coordinates"]["xmax"],
                pred["coordinates"]["ymax"],
            ),
        )
        for id, pred in predictions.items()
    ]


def _extract_seg_prediction(response: Dict[str, Any]) -> List[SegmentationPrediction]:
    """Extract Segmentation prediction result from response

    Parameters
    ----------
    response: Response from the LandingLens prediction endpoint.

    """
    encoded_predictions = response["backbonepredictions"]["bitmaps"]
    encoding_map = response["backbonepredictions"]["encoding"]["options"]["map"]
    mask_shape = (
        response["backbonepredictions"]["imageHeight"],
        response["backbonepredictions"]["imageWidth"],
    )
    return [
        SegmentationPrediction(
            id=id,
            label_name=bitmap_info["label_name"],
            label_index=bitmap_info["label_index"],
            score=bitmap_info["score"],
            encoded_mask=bitmap_info["bitmap"],
            encoding_map=encoding_map,
            mask_shape=mask_shape,
        )
        for id, bitmap_info in encoded_predictions.items()
    ]


def _extract_vp_prediction(response: Dict[str, Any]) -> List[SegmentationPrediction]:
    """Extract Visual Prompting result from response

    Parameters
    ----------
    response: Response from the LandingLens prediction endpoint.

    """
    encoded_predictions = response["predictions"]["bitmaps"]
    encoding_map = response["predictions"]["encoding"]["options"]["map"]
    mask_shape = (
        response["predictions"]["imageHeight"],
        response["predictions"]["imageWidth"],
    )
    return [
        SegmentationPrediction(
            id=id,
            label_name=bitmap_info["label_name"],
            label_index=bitmap_info["label_index"],
            score=bitmap_info["score"],
            encoded_mask=bitmap_info["bitmap"],
            encoding_map=encoding_map,
            mask_shape=mask_shape,
        )
        for id, bitmap_info in encoded_predictions.items()
    ]


PREDICTION_EXTRACTOR = {
    "ObjectDetectionPrediction": _extract_od_prediction,
    "SegmentationPrediction": _extract_seg_prediction,
    "ClassificationPrediction": _extract_class_prediction,
    "SegmentationPredictionVP": _extract_vp_prediction,
}
