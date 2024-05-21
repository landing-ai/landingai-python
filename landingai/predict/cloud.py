from landingai.common import (
    APIKey,
    ClassificationPrediction,
    InferenceMetadata,
    ObjectDetectionPrediction,
    Prediction,
    SegmentationPrediction,
)
from landingai.exceptions import RateLimitExceededError
from landingai.predict.utils import (
    PredictionExtractor,
    get_cloudinference_prediction,
    create_requests_session,
)
from landingai.telemetry import get_runtime_environment_info
from landingai.timer import Timer
from landingai.utils import load_api_credential, serialize_image


import PIL.Image
import numpy as np
from requests import Session
from tenacity import before_sleep_log, retry, retry_if_exception_type, wait_fixed

import json
import logging
import socket
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

_LOGGER = logging.getLogger(__name__)


class Predictor:
    """A class that calls your inference endpoint on the LandingLens platform."""

    _url: str = "https://predict.app.landing.ai/inference/v1/predict"
    _num_retry: int = 3
    _session: Session

    def __init__(
        self,
        endpoint_id: str,
        *,
        api_key: Optional[str] = None,
        check_server_ready: bool = True,
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
        check_server_ready : bool, optional
            Check if the cloud inference service is reachable, by default True
        """
        # Check if the cloud inference service is reachable
        if check_server_ready and not self._check_connectivity(url=self._url):
            raise ConnectionError(
                f"Failed to connect to the cloud inference service. Check that {self._url} is accesible from this device"
            )

        self._endpoint_id = endpoint_id
        self._api_credential = self._load_api_credential(api_key)
        extra_x_event = {
            "endpoint_id": self._endpoint_id,
            "model_type": "fast_and_easy",
        }
        headers = self._build_default_headers(self._api_credential, extra_x_event)
        self._session = create_requests_session(self._url, self._num_retry, headers)
        # performance_metrics keeps performance metrics for the last call to _do_inference()
        self._performance_metrics: Dict[str, int] = {}

    def _load_api_credential(self, api_key: Optional[str]) -> Optional[APIKey]:
        """
        Simple wrapper to load the API key from given string or env var.

        This wrapper is useful to allow subclasses of `Predictor` to override the behavior
        of loading the API key.
        For example: SnowflakeNativeAppPredictor doesn't use APIKey at all,
        so it can override this method to return None.
        """
        return load_api_credential(api_key)

    def _check_connectivity(
        self, url: Optional[str] = None, host: Optional[Tuple[str, int]] = None
    ) -> bool:
        if url:
            parsed_url = urlparse(url)
            if parsed_url.port:
                port = parsed_url.port
            elif parsed_url.scheme == "https":
                port = 443
            elif parsed_url.scheme == "http":
                port = 80
            else:
                port = socket.getservbyname(parsed_url.scheme)
            host = (parsed_url.hostname, port)  # type: ignore

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(host)  # type: ignore
        # print(f"Checking if {host[0]}:{host[1]} is open (res={result})")
        sock.close()
        return result == 0

    def _build_default_headers(
        self, api_key: Optional[APIKey], extra_x_event: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Build the HTTP headers for the request to the Cloud inference endpoint(s)."""
        tracked_properties = get_runtime_environment_info()
        if extra_x_event:
            tracked_properties.update(extra_x_event)
        tracking_data = {
            "event": "inference",
            "action": "POST",
            "properties": tracked_properties,
        }
        header = {
            "contentType": "multipart/form-data",
            "X-event": json.dumps(tracking_data),
        }
        if api_key is not None:
            header["apikey"] = api_key.api_key
        return header

    @retry(
        # All customers have a quota of images per minute. If the server return a 429, then we will wait 60 seconds and retry. Note that we will retry forever on 429s which is ok since the rate limiter will eventually allow the request to go through.
        retry=retry_if_exception_type(RateLimitExceededError),
        wait=wait_fixed(60),
        before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
    )
    @Timer(name="Predictor.predict")
    def predict(
        self,
        image: Union[np.ndarray, PIL.Image.Image],
        metadata: Optional[InferenceMetadata] = None,
        **kwargs: Any,
    ) -> List[Prediction]:
        """Call the inference endpoint and return the prediction result.

        Parameters
        ----------
        image
            The input image to be predicted. The image should be in the RGB format if it has three channels.
        metadata
            The (optional) metadata associated with this inference/image.
            Metadata is helpful for attaching additional information to the inference result so you can later filter the historical inference results by your custom values in LandingLens.

            See `landingai.common.InferenceMetadata` for more information.

        Returns
        -------
        The inference result in a list of dictionary
            Each dictionary is a prediction result.
            The inference result has been filtered by the confidence threshold set in LandingLens and sorted by confidence score in descending order.
        """
        buffer_bytes = serialize_image(image)
        files = {"file": buffer_bytes}
        query_params = {
            "endpoint_id": self._endpoint_id,
        }
        data = {"metadata": metadata.json()} if metadata else None
        (preds, self._performance_metrics) = get_cloudinference_prediction(
            self._session,
            self._url,
            files,
            query_params,
            _CloudPredictionExtractor,
            data=data,
        )
        return preds

    def get_metrics(self) -> Dict[str, int]:
        """
        Return the performance metrics for the last inference call.

        Returns:
            A dictionary containing the performance metrics.
            Example:
            {
                "decoding_s": 0.0084266,
                "infer_s": 3.3537345,
                "postprocess_s": 0.0255059,
                "preprocess_s": 0.0124037,
                "waiting_s": 0.0001487
            }
        """
        return self._performance_metrics


class _CloudPredictionExtractor(PredictionExtractor):
    """A class that extract the raw JSON inference result to Predict Results instances."""

    @staticmethod
    def _extract_class_prediction(
        response: Dict[str, Any],
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
        response: Dict[str, Any],
    ) -> List[ObjectDetectionPrediction]:
        """Extract Object Detection prediction result from response

        Parameters
        ----------
        response: Response from the LandingLens prediction endpoint.
        Example input:
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
        response: Dict[str, Any],
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
        response: Dict[str, Any],
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
            predictions = _CloudPredictionExtractor._extract_od_prediction(response)  # type: ignore
        elif response_type == "SegmentationPrediction":
            predictions = _CloudPredictionExtractor._extract_seg_prediction(response)  # type: ignore
        elif response_type == "ClassificationPrediction":
            predictions = _CloudPredictionExtractor._extract_class_prediction(response)  # type: ignore
        elif response_type == "SegmentationPredictionVP":
            predictions = _CloudPredictionExtractor._extract_vp_prediction(response)  # type: ignore
        else:
            raise NotImplementedError(
                f"{response_type} is not implemented in _Extractor"
            )
        return predictions
