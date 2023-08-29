from typing import Dict, Optional

from landingai.data_management.client import GET_DEFECTS, LandingLens


class Label:
    """Label management API client.
    This class provides a set of APIs to manage the label of a particular project on LandingLens.
    For example, you can use this class to list all the available labels for a given project.

    Example
    -------
    >>> client = Label(project_id, api_key)
    >>> client.get_label_map()
    >>> {'0': 'ok', '1': 'cat', '2': 'dog'}

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)

    def get_label_map(self) -> Dict[str, str]:
        """Get all the available labels for a given project.

        Returns
        ----------
        Dict[str, str]
            A dictionary of label index to label name.
            ```
            # Example output
            {
                "0": "ok",
                "1": "cat",
                "2": "dog",
                "3": "duck",
            }
            ```
        """
        project_id = self._client._project_id
        resp = self._client._api(GET_DEFECTS, params={"projectId": project_id})
        resp_data = resp["data"]
        label_map = {str(label["indexId"]): label["name"] for label in resp_data}
        label_map["0"] = "ok"
        return label_map
