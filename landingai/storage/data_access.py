import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from typing import Optional


from landingai.io import read_file

_LOGGER = logging.getLogger(__name__)


def download_public_file(
    url: str,
    filename: Optional[Path] = None,
) -> Path:
    """Download a file from a public url. This function will follow redirects

    Parameters
    ----------
    url : str
        Source url
    output_path : Optional[Path], optional
        The local output file path for the downloaded file. If no path is provided, the file will be saved into a temporary directory provided by the OS (which could get deleted after reboot), and when possible the extension of the downloaded file will be included in the output file path.

    Returns
    -------
    Path
        Path to the downloaded file
    """
    # TODO: It would be nice for this function to not re-download if the src has not been updated
    data_bytes, server_filename = read_file(url)  # Fetch the file
    if filename is not None:
        with open(str(filename), "wb") as f:
            f.write(data_bytes)

    else:
        with tempfile.NamedTemporaryFile(
            suffix=f"--{server_filename}", delete=False
        ) as f:
            f.write(data_bytes)
    return f.name


def get_local_uri(uri: str) -> Path:
    """Check if the URI is local and fetch it if it is not

    Parameters
    ----------
    uri : str
        Supported URIs
        - local paths
        - file://
        - http://
        - hhtps://


    Returns
    -------
    Path
        Path to a local resource
    """
    r = urlparse(uri)
    if r.scheme == "" or r.scheme == "file":
        # The file is already local
        return Path(uri)
    if r.scheme == "http" or r.scheme == "https":
        # Fetch the file from the web
        return Path(download_public_file(uri))
