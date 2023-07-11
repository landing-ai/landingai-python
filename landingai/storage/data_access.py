import logging
import tempfile
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse


from landingai.io import read_file

_LOGGER = logging.getLogger(__name__)


def download_file(
    url: str,
    file_output_path: Optional[Path] = None,
) -> str:
    """Download a file from a public url. This function will follow redirects

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
        with tempfile.NamedTemporaryFile(
            suffix="--" + str(ret["filename"]), delete=False
        ) as f:
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
    if r.scheme == "" or r.scheme == "file":
        # The file is already local
        return Path(uri)
    if r.scheme == "http" or r.scheme == "https":
        # Fetch the file from the web
        return Path(download_file(uri))
    raise ValueError(f"URI not supported {uri}")
