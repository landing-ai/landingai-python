import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import requests

_LOGGER = logging.getLogger(__name__)


# TODO: support output type stream
def read_file(url: str) -> Dict[str, Any]:
    """Read bytes from a URL.
    Typically, the URL is a presigned URL (for example, from Amazon S3 or Snowflake) that points to a video or image file.
    Returns
    -------
    Dict[str, Any]
        Returns the content under "content". Optionally may return "filename" in case the server provided it.
    """
    response = requests.get(url, allow_redirects=True)  # True is the default behavior
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        reason = f"{e.response.text} (status code: {e.response.status_code})"
        msg_prefix = f"Failed to read from url ({url}) due to {reason}"
        if response.status_code == 403:
            error_msg = f"{msg_prefix}. Please double check the url is not expired and it's well-formed."
            raise ValueError(error_msg) from e
        elif response.status_code == 404:
            raise FileNotFoundError(
                f"{msg_prefix}. Please double check the file exists and the url is well-formed."
            ) from e
        else:
            error_msg = f"{msg_prefix}. Please try again later or reach out to us via our LandingAI platform."
            raise ValueError(error_msg) from e
    if response.status_code >= 300:
        raise ValueError(
            f"Failed to read from url ({url}) due to {response.text} (status code: {response.status_code})"
        )
    ret = {"content": response.content}
    # Check if server returned the file name
    if "content-disposition" in response.headers:
        m = re.findall(
            "filename=[\"']*([^;\"']+)", response.headers["content-disposition"]
        )
        if len(m):  # if there is a match select the first one
            ret["filename"] = m[0]
    _LOGGER.info(
        f"Received content with length {len(response.content)}, type {response.headers.get('Content-Type')}"
        # and filename "+ str(ret["filename"])
    )

    return ret


def download_file(
    url: str,
    file_output_path: Optional[Path] = None,
) -> str:
    """Download a file from a public url. This function will follow HTTP redirects

    Parameters
    ----------
    url : str
        Source url
    file_output_path : Optional[Path], optional
        The local output file path for the downloaded file. If no path is provided, the file will be saved into a temporary directory provided by the OS (which could get deleted after reboot), and when possible the extension of the downloaded file will be included in the output file path.

    Returns
    -------
    Path
        Path to the downloaded file
    """
    # TODO: It would be nice for this function to not re-download if the src has not been updated
    ret = read_file(url)  # Fetch the file
    if file_output_path is not None:
        with open(str(file_output_path), "wb") as f:  # type: Any
            f.write(ret["content"])

    else:
        suffix = ""
        if "filename" in ret:
            # use filename provided by server
            suffix = "--" + str(ret["filename"])
        else:
            # try to get the name from the URL
            r = urlparse(url)
            suffix = "--" + os.path.basename(unquote(r.path))
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(ret["content"])
    return f.name  # type: ignore


def fetch_from_uri(uri: str, **kwargs) -> Path:  # type: ignore
    """Check if the URI is local and fetch it if it is not

    Parameters
    ----------
    uri : str
        Supported URIs
        - local paths
        - file://
        - http://
        - https://


    Returns
    -------
    Path
        Path to a local resource
    """
    # TODO support other URIs
    # snowflake://stage/filename  (credentials will be passed on kwargs)
    r = urlparse(uri)
    # Match local unix and windows paths (e.g. C:\)
    if r.scheme == "" or r.scheme == "file" or len(r.scheme) == 1:
        # The file is already local
        return Path(uri)
    if r.scheme == "http" or r.scheme == "https":
        # Fetch the file from the web
        return Path(download_file(uri))
    raise ValueError(f"URI not supported {uri}")
