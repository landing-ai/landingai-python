from typing import  Optional

from landingai.data_management.client import EVENT_LOGS, LandingLens
from datetime import datetime, timezone
import requests
import logging

_LOGGER = logging.getLogger(__name__)


class Exporter:
    """Export management API client.
    This class provides a set of APIs to export data from LandingLens.
    For example, you can use this class to export all the available event logs of your organization.

    Example
    -------
    >>> client = Exporter(project_id, api_key)
    >>> client.export_event_logs("2023-06-01", '/path/to/save/file.csv')
    >>> # The csv file will be saved in the desired path.

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)

    def export_event_logs(self, from_date: str, save_path: str) -> None:
        """Exports the event logs of the organization from the given time range.

        Parameters
        ----------
        from_date: str
            In date following the format "YYYY-MM-DD"
        save_path:
            Desired path to save the csv file to: '/path/to/save/file.csv'

        Returns
        ----------
        None
            The csv file will be saved in the desired path.
        """
        try:
            datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("from_date must be in YYYY-MM-DD format")
        from_timestamp = datetime.strptime(f"{from_date} 00:00:00.00000", "%Y-%m-%d %H:%M:%S.%f")
        to_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

        _LOGGER.info("Exporting event logs...")
        resp = self._client._api(EVENT_LOGS, params={"fromTimestamp": from_timestamp, "toTimestamp": to_timestamp})
        signed_url = resp["data"].get("signedUrl")
        _LOGGER.debug("Signed URL: ", signed_url)
        self._download_file_from_signed_url(signed_url, save_path)
        print(f"Event logs exported successfully to path: {save_path}")
        return

    def _download_file_from_signed_url(self, signed_url: str, save_path: str):
        response = requests.get(signed_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
