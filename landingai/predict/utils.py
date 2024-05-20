"""Module for making predictions on LandingLens models."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from landingai.common import (
    Prediction,
)
from landingai.exceptions import HttpResponse
from landingai.timer import Timer

_LOGGER = logging.getLogger(__name__)


def serialize_rois(rois: List[List[Tuple[int, int]]], mode: str) -> str:
    """Serialize the regions of interest into a JSON string."""
    rois_payload = [
        {
            "location": [{"x": coord[0], "y": coord[1]} for coord in roi],
            "mode": mode,
        }
        for roi in rois
    ]
    return json.dumps([rois_payload])


class PredictionExtractor:
    """The base class for all extractors. This is useful for type checking."""

    @staticmethod
    def extract_prediction(response: Any) -> List[Prediction]:
        raise NotImplementedError()


def create_requests_session(
    url: str, num_retry: int, headers: Dict[str, str]
) -> Session:
    """Create a requests session with retry"""
    session = Session()
    retries = Retry(
        # TODO: make them configurable
        # The 5XX retry scheme needs to account for the circuit breaker which will shutdown a service for 10 seconds
        total=num_retry,  # Defaults to 3
        backoff_factor=7,  # This the amount of seconds to wait on the second retry (i.e. 0, 7, 21). First retry is immediate.
        raise_on_redirect=True,
        raise_on_status=False,  # We are already raising exceptions during backend invocations
        allowed_methods=["GET", "POST", "PUT"],
        status_forcelist=[
            # 408 Request Timeout , 413 Content Too Large
            # 429,  # Too Many Requests  (ie. rate limiter). This is handled externally
            # 500 Internal Server Error -> We don't retry here since it tends to reflect determinist software bugs
            502,  # Bad Gateway
            503,  # Service Unavailable (include cloud circuit breaker)
            504,  # Gateway Timeout
        ],
    )
    session.mount(
        url, HTTPAdapter(max_retries=retries if num_retry > 0 else num_retry)
    )  # Since POST is not idempotent we will ony retry on the this specific API
    session.headers.update(headers)
    return session


@Timer(name="_do_inference", log_fn=_LOGGER.debug)
def get_cloudinference_prediction(
    session: Session,
    endpoint_url: str,
    files: Dict[str, Any],
    params: Dict[str, Any],
    extractor_class: Type[PredictionExtractor],
    *,
    data: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Prediction], Dict[str, int]]:
    """Call the inference endpoint and extract the prediction result."""
    try:
        resp = session.post(endpoint_url, files=files, params=params, data=data)
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to the model server. Please double check the model server url ({endpoint_url}) is correct.\nException detail: {e}"
        ) from e
    response = HttpResponse.from_response(resp)
    _LOGGER.debug("Response: %s", response)
    response.raise_for_status()
    json_dict = response.json()
    # OCR response is a list of list of predictions
    if isinstance(json_dict, list):
        return (extractor_class.extract_prediction(json_dict), {})
    # Save performance metrics for debugging
    performance_metrics = json_dict.get("latency", {})
    return (extractor_class.extract_prediction(json_dict), performance_metrics)
