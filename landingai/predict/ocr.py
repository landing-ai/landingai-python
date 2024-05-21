from tenacity import before_sleep_log, retry, retry_if_exception_type, wait_fixed
from landingai.common import OcrPrediction, Prediction
from landingai.exceptions import RateLimitExceededError
from landingai.predict.cloud import Predictor
from landingai.predict.utils import (
    PredictionExtractor,
    get_cloudinference_prediction,
    create_requests_session,
    serialize_rois,
)
from landingai.timer import Timer
from landingai.utils import load_api_credential, serialize_image


import PIL.Image
import numpy as np


import logging
from typing import Any, Dict, List, Literal, Optional, Union, cast


_LOGGER = logging.getLogger(__name__)


class OcrPredictor(Predictor):
    """A class that calls your OCR inference endpoint on the LandingLens platform."""

    _url: str = "https://app.landing.ai/ocr/v1/detect-text"

    def __init__(
        self,
        threshold: float = 0.5,
        *,
        language: Literal["en", "ch"] = "ch",
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
        language:
            Specifies the character set to use. Can either be `"en"` for English
            or `"ch"` for Chinese and English (default).
        """
        self._threshold = threshold
        self._language = language
        self._api_credential = load_api_credential(api_key)
        extra_x_event = {
            "model_type": "ocr",
        }
        headers = self._build_default_headers(self._api_credential, extra_x_event)
        self._session = create_requests_session(self._url, self._num_retry, headers)

    @retry(
        # All customers have a quota of images per minute. If the server return a 429, then we will wait 60 seconds and retry
        retry=retry_if_exception_type(RateLimitExceededError),
        wait=wait_fixed(60),
        before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
    )
    @Timer(name="OcrPredictor.predict")
    def predict(  # type: ignore
        self, image: Union[np.ndarray, PIL.Image.Image], **kwargs: Any
    ) -> List[Prediction]:
        """Run OCR on the input image and return the prediction result.

        Parameters
        ----------
        image:
            The input image to be predicted
        mode:
            The mode of this prediction. It can be either "multi-text" (default) or "single-text".
            In "multi-text" mode, the predictor will detect multiple lines of text in the image.
            In "single-text" mode, the predictor will detect a single line of text in the image.
        regions_of_interest:
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
        files = {"images": buffer_bytes}
        mode: str = kwargs.get("mode", "multi-text")
        if mode not in ["multi-text", "single-text"]:
            raise ValueError(
                f"mode must be either 'multi-text' or 'single-text', but got: {mode}"
            )
        if mode == "single-text" and "regions_of_interest" not in kwargs:
            raise ValueError(
                "regions_of_interest parameter must be provided in single-text mode."
            )
        data: Dict[str, Any]
        data = {"language": self._language}
        if rois := kwargs.get("regions_of_interest", []):
            data["rois"] = serialize_rois(rois, mode)

        (preds, self._performance_metrics) = get_cloudinference_prediction(
            self._session,
            self._url,
            files,
            {},
            _OcrExtractor,
            data=data,
        )
        return [pred for pred in preds if pred.score >= self._threshold]


class _OcrExtractor(PredictionExtractor):
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
