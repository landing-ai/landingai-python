"""Module that defines all the exceptions used in the package, and APIs for request/response error handling."""

import logging
from pprint import pformat
from typing import Any, Dict, List, Union, cast

from requests import PreparedRequest, Response
from requests.structures import CaseInsensitiveDict

_LOGGER = logging.getLogger(__name__)


class InvalidApiKeyError(Exception):
    """Exception raised when the an invalid API key is provided. This error could be raised from any SDK code, not limited to a HTTP client."""

    def __init__(self, message: str):
        self.message = f"""{message}
For more information, see https://landing-ai.github.io/landingai-python/landingai.html#manage-api-credentials"""
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class UnauthorizedError(Exception):
    """Exception raised when the user is not authorized to access the resource. Status code: 401."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class BadRequestError(Exception):
    """Exception raised when the request is not invalid. It could be due to that the Endpoint ID doesn't exist. Status code: 400, 413, 422 or other 4xx."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class PermissionDeniedError(Exception):
    """Exception raised when the requested action is not allowed. It could be due to that the user runs out of credits or the enterprise contract expired, etc. Status code: 403."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class RateLimitExceededError(Exception):
    """Exception raised when the inference request is rate limited. Status code: 429."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ServiceUnavailableError(Exception):
    """Exception raised when the requested service is unavailable temporarily. Status code: 502, 503, 504."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InternalServerError(Exception):
    """Exception raised when the server encounters an unexpected condition that prevents it from fulfilling the request. Status code: 5xx."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class ClientError(Exception):
    """Exception raised when the server failed due to client errors. Status code: 4xx."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class HttpError(Exception):
    """Exception raised when there's something wrong with the HTTP request. This is a generic exception that is raised when no other more specific exception is appropriate. Status code: 3xx, 4xx or 5xx."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class DuplicateUploadError(Exception):
    """Exception raised when the uploaded media is already exists in the project. Status code: 409."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class UnexpectedRedirectError(Exception):
    """Exception raised when the client encounters an unexpected redirect. Status code: 3xx."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class HttpResponse:
    """Abstraction for the HTTP response from the prediction endpoint.
    It encapulates more granular and friendly error handling logic that tailored to LandingLens.
    """

    def __init__(
        self,
        status_code: int,
        reason: str,
        uri: str,
        data: Union[str, Dict[str, Any]],
        headers: CaseInsensitiveDict,
        content_length: int,
        request: PreparedRequest,
    ) -> None:
        self.status_code = status_code
        self.reason = reason
        self.uri = uri
        self.data = data
        self.headers = headers
        self.content_length = content_length
        self.request = request

    def __str__(self) -> str:
        return pformat(self.__dict__)

    def json(self) -> Union[Dict[str, Any], List[Any]]:
        """Get json body from the response.
        NOTE: make sure you have checked the response and you are expecting a JSON body in response

        Returns
        -------
        Union[Dict[str, Any], List[Any]]
            the json body of the response. Currently, it could be either a dict or a list. (It may support more types in the future)
        """
        assert isinstance(self.data, dict) or isinstance(
            self.data, list
        ), f"expecting a dict or list instance, but it's f{type(self.data)} typed."
        return self.data

    @classmethod
    def from_response(cls, response: Response) -> "HttpResponse":
        """Create an instance of _HttpResponse from a requests.Response object."""
        try:
            content: Union[str, Dict[str, Any]] = response.json()
        except ValueError:
            # In some circumstance, the endpoint may return string content instead.
            content = str(response.content, "utf-8")
        return cls(
            status_code=response.status_code,
            reason=response.reason,
            uri=response.url,
            data=content,
            headers=cast(CaseInsensitiveDict, response.headers),
            content_length=len(response.content),
            request=response.request,
        )

    def raise_for_status(self) -> None:
        """Raises an exception if the HTTP status is erroneous.
        This method is more granular than the one provided by requests.Response.
        It's tailored to LandingLens model inference endpoints.
        """
        if self.is_2xx:
            return
        response_detail_str = f"Response detail:\n{self}"
        _LOGGER.error(
            f"Request failed with status code {self.status_code}: {self.reason}\n{response_detail_str}"
        )
        if self.status_code == 401:
            raise UnauthorizedError(
                f"Unauthorized. Please check your API key and API secret is correct.\n{response_detail_str}"
            )
        elif self.status_code == 403:
            raise PermissionDeniedError(
                f"Permission denied. Please check your account has enough credits or your enterprise contract is not expired or if you have access to this endpoint. Contact your account admin for more information.\n{response_detail_str}"
            )
        elif self.status_code == 404:
            raise BadRequestError(
                f"Endpoint doesn't exist. Please check the inference url path and other configuration is correct and try again. If this issue persists, please report this issue to LandingLens support for further assistant.\n{response_detail_str}"
            )
        elif self.status_code == 422:
            raise BadRequestError(
                f"Bad request. Please check your Endpoint ID is correct.\n{response_detail_str}"
            )
        elif self.status_code == 429:
            raise RateLimitExceededError(
                f"Rate limit exceeded. You have sent too many requests in a minute. Please wait for a minute before sending new requests. Contact your account admin or LandingLens support for how to increase your rate limit.\n{response_detail_str}"
            )
        elif self.status_code in [502, 503, 504]:
            raise ServiceUnavailableError(
                f"Service temporarily unavailable. Please try again in a few minutes.\n{response_detail_str}"
            )
        elif self.is_3xx:  # not expecting any 3xx response
            raise UnexpectedRedirectError(
                "Unexpected redirect. Please report this issue to LandingLens support for further assistant."
            )
        elif self.is_4xx:  # other 4xx errors
            raise ClientError(
                f"Client error. Please check your configuration and inference request is well-formed and try again. If this issue persists, please report this issue to LandingLens support for further assistant.\n{response_detail_str}"
            )
        elif self.is_5xx:  # other 5xx errors
            raise InternalServerError(
                f"Internal server error. The model server encountered an unexpected condition and failed. Please report this issue to LandingLens support for further assistant.\n{response_detail_str}"
            )
        raise AssertionError(
            f"Unexpected status code: {self.status_code}. Please report this issue to LandingLens support.\n{response_detail_str}"
        )

    @property
    def is_2xx(self) -> bool:
        return self.status_code < 300

    @property
    def is_3xx(self) -> bool:
        return 300 <= self.status_code < 400

    @property
    def is_4xx(self) -> bool:
        return 400 <= self.status_code < 500

    @property
    def is_5xx(self) -> bool:
        return 500 <= self.status_code < 600
