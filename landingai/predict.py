"""Module for making predictions on LandingLens models."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import PIL.Image
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from landingai.common import (
    APIKey,
    ClassificationPrediction,
    ObjectDetectionPrediction,
    OcrPrediction,
    Prediction,
    SegmentationPrediction,
)
from landingai.exceptions import HttpResponse
from landingai.telemetry import get_runtime_environment_info, is_running_in_pytest
from landingai.utils import load_api_credential, serialize_image

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """A class that calls your inference endpoint on the LandingLens platform."""

    _url: str = "https://predict.app.landing.ai/inference/v1/predict"
    _num_retry: int = 3

    def __init__(
        self,
        endpoint_id: str,
        *,
        api_key: Optional[str] = None,
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
        """
        self._endpoint_id = endpoint_id
        self._api_credential = load_api_credential(api_key)
        headers = self._build_default_headers(self._api_credential)
        self._session = _create_session(Predictor._url, self._num_retry, headers)

    def _build_default_headers(self, api_key: APIKey) -> Dict[str, str]:
        """Build the HTTP headers for the request to the Cloud inference endpoint(s)."""
        headers = {
            "contentType": "application/json",
            "apikey": api_key.api_key,
        }
        return headers

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
        buffer_bytes = serialize_image(image)
        files = [("file", ("image.png", buffer_bytes, "image/png"))]
        payload = {
            "endpoint_id": self._endpoint_id,
            "device_type": "pylib",
        }
        _add_defualt_query_params(payload)
        return _do_inference(
            self._session, Predictor._url, files, payload, _CloudExtractor
        )


class OcrPredictor(Predictor):
    """A class that calls your OCR inference endpoint on the LandingLens platform."""

    _url: str = "https://app.staging.landing.ai/ocr/v1/detect-text"

    def __init__(
        self,
        threshold: float = 0.5,
        *,
        api_key: Optional[str] = None,
    ) -> None:
        """OCR Predictor constructor

        Parameters
        ----------
        threshold:
            The minimum confidence threshold of the prediction to keep, by default 0.5
        api_key
            The API Key of your LandingLens organization.
            If not provided, it will try to load from the environment variable
            LANDINGAI_API_KEY or from the .env file.
        """
        self._threshold = threshold
        self._api_credential = load_api_credential(api_key)
        headers = self._build_default_headers(self._api_credential)
        self._session = _create_session(Predictor._url, self._num_retry, headers)

    def predict(
        self, image: Union[np.ndarray, PIL.Image.Image], **kwargs: Any
    ) -> List[Prediction]:
        """Run OCR on the input image and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted
        mode:
            The mode of this prediction. It can be either "multi-text" (default) or "single-text".
            In "multi-text" mode, the predictor will detect multiple lines of text in the image.
            In "single-text" mode, the predictor will detect a single line of text in the image.
        regions_of_interest
            A list of region of interest boxes/quadrilateral. Each quadrilateral is a list of 4 points (x, y).
            In "single-text" mode, the caller must provide a list of quadrilateral(s) that cover the text in the image.
            Each quadrilateral is a list of 4 points (x, y), and it should cover a single line of text in the image.
            In "multi-text" mode, regions_of_interest is not required. If it is None, the whole image will be used as the region of interest.

        Returns
        -------
        List[OcrPrediction]
            A list of OCR prediction result.
        """

        buffer_bytes = serialize_image(image)
        files = [("images", ("image.png", buffer_bytes, "image/png"))]
        mode: str = kwargs.get("mode", "multi-text")
        if mode not in ["multi-text", "single-text"]:
            raise ValueError(
                f"mode must be either 'multi-text' or 'single-text', but got: {mode}"
            )
        if mode == "single-text" and "regions_of_interest" not in kwargs:
            raise ValueError(
                "regions_of_interest parameter must be provided in single-text mode."
            )
        data = {}
        rois: List[List[Tuple[int, int]]] = kwargs.get("regions_of_interest", [])
        if rois:
            data["rois"] = serialize_rois(rois, mode)

        payload: Dict[str, Any] = {"device_type": "pylib"}
        _add_defualt_query_params(payload)
        preds = _do_inference(
            self._session,
            OcrPredictor._url,
            files,
            payload,
            _OcrExtractor,
            data=data,
        )
        return [pred for pred in preds if pred.score >= self._threshold]


def serialize_rois(rois: List[List[Tuple[int, int]]], mode: str) -> str:
    """Serialize the regions of interest into a JSON string."""
    rois_payload = []
    for roi in rois:
        rois_payload.append(
            {
                "location": [{"x": coord[0], "y": coord[1]} for coord in roi],
                "mode": mode,
            }
        )
    return json.dumps([rois_payload])


class EdgePredictor(Predictor):
    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        self._url = f"http://{host}:{port}/images"
        self._session = _create_session(
            self._url, self._num_retry, {"contentType": "multipart/form-data"}
        )

    def predict(
        self, image: Union[np.ndarray, PIL.Image.Image], **kwargs: Any
    ) -> List[Prediction]:
        """Run Edge inference on the input image and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted

        Returns
        -------
        List[Prediction]
            A list of prediction result.
        """
        buffer_bytes = serialize_image(image)
        files = [("file", ("image.png", buffer_bytes, "image/png"))]
        return _do_inference(self._session, self._url, files, {}, _EdgeExtractor)


class _Extractor:
    """The base class for all extractors. This is useful for type checking."""

    @staticmethod
    def extract_prediction(response: Any) -> List[Prediction]:
        raise NotImplementedError()


class _CloudExtractor(_Extractor):
    """A class that extract the raw JSON inference result to Predict Results instances."""

    @staticmethod
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

    @staticmethod
    def _extract_od_prediction(
        response: Dict[str, Any]
    ) -> List[ObjectDetectionPrediction]:
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

    @staticmethod
    def _extract_seg_prediction(
        response: Dict[str, Any]
    ) -> List[SegmentationPrediction]:
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

    @staticmethod
    def _extract_vp_prediction(
        response: Dict[str, Any]
    ) -> List[SegmentationPrediction]:
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

    @staticmethod
    def extract_prediction(response: Dict[str, Any]) -> List[Prediction]:
        response_type = response.get("backbonetype")
        if response_type is None and response["type"] == "SegmentationPrediction":
            response_type = "SegmentationPredictionVP"  # Visual Prompting response
        if response_type is None:
            response_type = response["type"]  # Classification response
        predictions: List[Prediction]
        if response_type == "ObjectDetectionPrediction":
            predictions = _CloudExtractor._extract_od_prediction(response)  # type: ignore
        elif response_type == "SegmentationPrediction":
            predictions = _CloudExtractor._extract_seg_prediction(response)  # type: ignore
        elif response_type == "ClassificationPrediction":
            predictions = _CloudExtractor._extract_class_prediction(response)  # type: ignore
        elif response_type == "SegmentationPredictionVP":
            predictions = _CloudExtractor._extract_vp_prediction(response)  # type: ignore
        else:
            raise NotImplementedError(
                f"{response_type} is not implemented in _Extractor"
            )
        return predictions


class _EdgeExtractor(_Extractor):
    """A class that extract the raw Edge JSON inference result to Predict Results instances."""

    @staticmethod
    def _extract_edge_class_prediction(
        response: Dict[str, Any]
    ) -> List[ClassificationPrediction]:
        """Extract Edge Inference Classification prediction result from response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.
        Example input:
        {
            "type": "ClassificationPrediction",
            "predictions":
            {
                "score": 0.9951885938644409,
                "labelIndex": 0,
                "labelName": "Fire"
            }
        }
        """
        return [
            ClassificationPrediction(
                score=response["predictions"]["score"],
                label_index=response["predictions"]["labelIndex"],
                label_name=response["predictions"]["labelName"],
            )
        ]

    @staticmethod
    def _extract_edge_od_prediction(
        response: Dict[str, Any]
    ) -> List[ObjectDetectionPrediction]:
        """Extract Object Detection prediction result from edge inference response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.
        Example example input:
        {
            "type": "ObjectDetectionPrediction",
            "predictions":
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
            }
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
        predictions = response["predictions"]
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

    @staticmethod
    def _extract_edge_seg_prediction(
        response: Dict[str, Any]
    ) -> List[SegmentationPrediction]:
        """Extract Segmentation prediction result from response

        Parameters
        ----------
        response: Response from the Edge prediction endpoint.

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
                label_name=bitmap_info["labelName"],
                label_index=bitmap_info["labelIndex"],
                score=bitmap_info["score"],
                encoded_mask=bitmap_info["bitmap"],
                encoding_map=encoding_map,
                mask_shape=mask_shape,
            )
            for id, bitmap_info in encoded_predictions.items()
        ]

    @staticmethod
    def extract_prediction(response: Dict[str, Any]) -> List[Prediction]:
        response_type = response.get("type")
        if response_type is None:
            response_type = response["type"]
        predictions: List[Prediction]
        if response_type == "ObjectDetectionPrediction":
            predictions = _EdgeExtractor._extract_edge_od_prediction(response)  # type: ignore
        elif response_type == "SegmentationPrediction":
            predictions = _EdgeExtractor._extract_edge_seg_prediction(response)  # type: ignore
        elif response_type == "ClassificationPrediction":
            predictions = _EdgeExtractor._extract_edge_class_prediction(response)  # type: ignore
        else:
            raise NotImplementedError(
                f"{response_type} is not implemented in EdgeExtractor"
            )
        return predictions


class _OcrExtractor(_Extractor):
    """A class that extract the raw JSON inference result to OcrPrediction instances."""

    @staticmethod
    def extract_prediction(response: List[List[Dict[str, Any]]]) -> List[Prediction]:
        """Extract a batch of OCR prediction results from the response."""
        preds = [
            OcrPrediction(
                text=pred["text"],
                location=[(coord["x"], coord["y"]) for coord in pred["location"]],
                score=pred["score"],
            )
            for pred in response[0]  # batch size is always 1
        ]
        return cast(List[Prediction], preds)


def _create_session(url: str, num_retry: int, headers: Dict[str, str]) -> Session:
    """Create a requests session with retry"""
    session = Session()
    retries = Retry(
        # TODO: make them configurable
        total=num_retry,
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
        url, HTTPAdapter(max_retries=retries)
    )  # Since POST is not idempotent we will ony retry on the this specific API
    session.headers.update(headers)
    return session


def _do_inference(
    session: Session,
    endpoint_url: str,
    files: List[Any],
    payload: Dict[str, Any],
    extractor_class: Type[_Extractor],
    *,
    data: Optional[Dict[str, Any]] = None,
) -> List[Prediction]:
    """Call the inference endpoint and extract the prediction result."""
    try:
        resp = session.post(endpoint_url, files=files, params=payload, data=data)
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to the model server. Please double check the model server url ({endpoint_url}) is correct.\nException detail: {e}"
        )
    response = HttpResponse.from_response(resp)
    _LOGGER.debug("Response: %s", response)
    response.raise_for_status()
    json_dict = response.json()
    return extractor_class.extract_prediction(json_dict)


def _add_defualt_query_params(payload: Dict[str, Any]) -> None:
    """Add default query params to the payload for tracking and analystics purppose."""
    env_info = get_runtime_environment_info()
    payload["runtime"] = env_info["runtime"]
    if is_running_in_pytest():
        # Don't add extra query params if pytest is running, otherwise it will fail some unit tests
        return
    payload["lib_version"] = env_info["lib_version"]
