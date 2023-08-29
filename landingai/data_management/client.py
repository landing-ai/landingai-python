import io
import json
import logging
import posixpath
from importlib.metadata import version
from typing import Any, Dict, Optional, Tuple, cast

import aiohttp
import requests

from landingai.data_management.utils import to_camel_case
from landingai.exceptions import HttpError
from landingai.utils import load_api_credential

METADATA_ITEMS = "metadata_items"
METADATA_UPDATE = "metadata_update"
MEDIA_LIST = "media_list"
MEDIA_REGISTER = "media_register"
MEDIA_SIGN = "media_sign"
MEDIA_UPDATE_SPLIT = "media_update_split"
GET_PROJECT_SPLIT = "get_project_split"
GET_PROJECT = "get_project"
GET_DEFECTS = "get_defects"


ROUTES = {
    GET_PROJECT_SPLIT: {
        "root_url": "LANDING_API",
        "endpoint": "api/project/split",
        "method": requests.get,
    },
    MEDIA_UPDATE_SPLIT: {
        "root_url": "LANDING_API",
        "endpoint": "api/dataset/update_media_split",
        "method": requests.post,
    },
    GET_DEFECTS: {
        "root_url": "LANDING_API",
        "endpoint": "api/defect/defects",
        "method": requests.get,
    },
    METADATA_ITEMS: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/metadata/get_metadata_by_projectId",
        "method": requests.get,
    },
    METADATA_UPDATE: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/object/medias_metadata",
        "method": requests.post,
    },
    MEDIA_REGISTER: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/medias/new",
        "method": requests.post,
    },
    MEDIA_SIGN: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/medias/sign",
        "method": requests.post,
    },
    MEDIA_LIST: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/dataset/medias",
        "method": requests.get,
    },
    GET_PROJECT: {
        "root_url": "LANDING_API",
        "endpoint": "api/{version}/project/with_users",
        "method": requests.get,
    },
}

_URL_ROOTS = {
    "LANDING_API": "https://app.landing.ai",
}
_API_VERSION = "v1"
_LOGGER = logging.getLogger(__name__)


# Backward incompatible changes compared to LandingLens CLI:
# 1. Remove support for API key v1
# 2. Remove support for ~/.landinglens/config.ini file
# 3. Remove the no-pagination parameter in media.ls()
# 4. Rename upload() to update() in metadata.py


class LandingLens:
    """
    LandingLens client

    Example
    -------
    # Create a client by specifying API Key and project id
    >>> client = LandingLens(project, api_key)

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self.project_id = project_id
        if not api_key:
            api_key = load_api_credential().api_key
        self.api_key = api_key

    @property
    def _project_id(self) -> int:
        return self.project_id

    @property
    def _api_key(self) -> str:
        return self.api_key

    async def _api_async(
        self,
        route_name: str,
        params: Optional[Dict[str, Any]] = None,
        resp_with_content: Optional[Dict[str, Any]] = None,
        url_replacements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Returns a response from the LandingLens API"""
        assert resp_with_content is not None
        endpoint, headers, params, root_url, route = self._api_common_setup(
            resp_with_content, params, route_name, url_replacements
        )

        async with aiohttp.ClientSession() as session:
            # TODO: should be library agnostic
            if route["method"] == requests.get:
                method = session.get
            elif route["method"] == requests.post:
                method = session.post
            else:
                raise NotImplementedError()

            async with method(
                endpoint,
                headers=headers,
                json=resp_with_content,
                params=params,
                ssl=True,
            ) as resp:
                _LOGGER.debug("Request URL: ", resp.request_info.url)
                _LOGGER.debug("Response Code: ", resp.status)
                _LOGGER.debug("Response Reason: ", resp.reason)

                if resp.status != 200:
                    try:
                        content = await resp.text()
                        error_message = json.load(io.StringIO(content))["message"]
                    except Exception as e:
                        error_message = e
                    raise HttpError(
                        "HTTP request to LandingLens server failed with "
                        f"code {resp.status}-{resp.reason} and error message: \n"
                        f"{error_message}"
                    )
                resp_with_content = await resp.json()
                _LOGGER.debug(
                    "Response Content (500 chars): ",
                    json.dumps(resp_with_content)[:500],
                )
                assert resp_with_content is not None
                return resp_with_content

    def _api(
        self,
        route_name: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        url_replacements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Returns a response from the LandingLens API"""
        endpoint, headers, params, root_url, route = self._api_common_setup(
            data, params, route_name, url_replacements
        )
        resp = route["method"](
            endpoint,
            params=params,
            json=data,
            headers=headers,
            verify=True,
        )
        _LOGGER.debug("Request URL: ", resp.request.url)
        _LOGGER.debug("Request Body: ", resp.request.body)
        _LOGGER.debug("Response Code: ", resp.status_code)
        _LOGGER.debug("Response Reason: ", resp.reason)
        _LOGGER.debug("Response Content (500 chars): ", resp.content[:500])
        if not resp.ok:
            try:
                error_message = json.load(io.StringIO(resp.content.decode("utf-8")))[
                    "message"
                ]
            except Exception as e:
                error_message = e
            raise HttpError(
                "HTTP request to LandingLens server failed with "
                f"code {resp.status_code}-{resp.reason} and error message: \n"
                f"{error_message}"
            )
        return cast(Dict[str, Any], resp.json())

    def _api_common_setup(
        self,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        route_name: str,
        url_replacements: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], str, Dict[str, Any]]:
        route = ROUTES[route_name]
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "landingai-python-" + version("landingai"),
        }
        root_url_type = cast(str, route["root_url"])

        if root_url_type not in _URL_ROOTS:
            raise ValueError(f"Unknown URL specified: {root_url_type}")

        root_url = _URL_ROOTS[root_url_type]

        if not params:
            params = {}
        if route["method"] == requests.get and not params.get("projectId"):
            params["projectId"] = self.project_id
        if route["method"] == requests.post and data and not data.get("projectId"):
            data["projectId"] = self.project_id
        endpoint = posixpath.join(root_url, cast(str, route["endpoint"]))

        if url_replacements:
            endpoint = endpoint.format(
                **{**{"version": _API_VERSION}, **url_replacements}
            )
        else:
            endpoint = endpoint.format(**{"version": _API_VERSION})

        return endpoint, headers, params, root_url, route

    def get_project_property(
        self, project_id: int, property: Optional[str] = None
    ) -> Any:
        resp = self._api(GET_PROJECT, params={"projectId": project_id})
        project = resp.get("data")
        if property is None:
            return project
        assert project is not None
        property_value = project.get(to_camel_case(property))
        if property_value is None:
            raise HttpError(f"{property} Id not found")
        return property_value

    def get_metadata_mappings(
        self, project_id: int
    ) -> Tuple[Dict[str, Any], Dict[int, str]]:
        resp = self._api(METADATA_ITEMS, params={"projectId": project_id})
        metadata_mapping_resp = resp.get("data", {})

        metadata_mapping = {
            metadata_field["name"]: (
                metadata_field["id"],
                metadata_field["predefinedChoices"],
            )
            for metadata_field in metadata_mapping_resp.values()
        }
        id_to_metadata = {v[0]: k for k, v in metadata_mapping.items()}

        return metadata_mapping, id_to_metadata
