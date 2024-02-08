from typing import Dict, Optional

from landingai.data_management.client import EVENT_LOGS, LandingLens
from datetime import datetime


class Exporter:
    """Export management API client.
    This class provides a set of APIs to export data from LandingLens.
    For example, you can use this class to export all the available event logs of your organization.

    Example
    -------
    >>> client = Exporter(project_id, api_key)
    >>> client.export_event_logs("2023-06-01 00:00:00.000", "2023-12-30 00:00:00.000")
    >>> {'s3Path': 's3://.../1685595600-1703912400.csv', 'signedUrl': ''https://landinglens-bucket.s3....'}

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)

    def export_event_logs(self, from_timestamp: str, to_timestamp: str) -> Dict[str, str]:
        """Exports the event logs of the organization from the given time range.
        Parameters
        ----------
        from_timestamp: str
            In UTC following the format "YYYY-MM-DD HH:MM:SS.SSS"
        to_timestamp: str
            In UTC following the format "YYYY-MM-DD HH:MM:SS.SSS"

        Returns
        ----------
        Dict[str, str]
            A dictionary with the S3 path and signed url pointing to a downloadable csv file, that contains the event logs.
            ```
            # Example output
            {
                's3Path': 's3://.../1685595600-1703912400.csv',
                'signedUrl': ''https://landinglens-bucket.s3....'
            }
            ```
        """
        # Check if from_timestamp and to_timestamp are in UTC format
        try:
            datetime.strptime(from_timestamp, "%Y-%m-%d %H:%M:%S.%f")
            datetime.strptime(to_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            raise ValueError("from_timestamp and to_timestamp must be in UTC format")

        resp = self._client._api(EVENT_LOGS, params={"fromTimestamp": from_timestamp, "toTimestamp": to_timestamp})
        resp_data = resp["data"]
        return resp_data
