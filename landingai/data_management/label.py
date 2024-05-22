from typing import Dict, Optional

from landingai.data_management.client import GET_DEFECTS, CREATE_DEFECTS, LandingLens
from landingai.data_management.types.label import LabelType
from landingai.data_management.types.classes import ClassMap


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
        project_type = self._client.get_project_property(project_id, "labelType")
        resp = self._client._api_async(GET_DEFECTS, params={"projectId": project_id})
        resp_data = resp["data"]
        label_map = {str(label["index"]): label["name"] for label in resp_data}
        if project_type != LabelType.classification:
            label_map["0"] = "ok"
        return label_map

    def create_label_map(self, label_map: Dict[str, ClassMap]) -> Dict[str, str]:
        """Create labels into the selected project.
        Parameters
        ----------
        label_map: Dict[str, ClassMap]
            The label maps to be created. The key is the label index and the value is the label name.
            # Example input
            {"1": {"name": "Screw"},"2": {"name": "dust"}}

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
        project_type = self._client.get_project_property(project_id, "labelType")
        resp = self._client._api_async(
            CREATE_DEFECTS,
            resp_with_content=label_map,
        )
        resp_data = resp["data"]
        resp_label_map = {str(label["index"]): label["name"] for label in resp_data}
        if project_type != LabelType.classification:
            resp_label_map["0"] = "ok"
        return resp_label_map
