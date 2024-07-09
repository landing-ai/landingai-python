from typing import Any, Dict, List, Optional, Union

from landingai.data_management.client import METADATA_GET, METADATA_UPDATE, LandingLens
from landingai.data_management.utils import (
    PrettyPrintable,
    ids_to_metadata,
    metadata_to_ids,
    obj_to_dict,
)


class Metadata:
    """Metadata management API client.
    This class provides a set of APIs to manage the metadata of the medias (images) uploaded to LandingLens.
    For example, you can use this class to update the metadata of the uploaded medias.

    Example
    -------
    >>> client = Metadata(project_id, api_key)
    >>> client.update([101, 102, 103], creator="tom")

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)

    def update(
        self,
        media_ids: Union[int, List[int]],
        **input_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update or insert a dictionary of metadata for a set of medias.

        Parameters
        ----------
        media_ids
            Media ids to update.
        input_metadata
            A dictionary of metadata to be updated or inserted. The key of the metadata
            needs to be created/registered (for the first time) on LandingLens before
            calling update().

        Returns
        ----------
        Dict[str, Any]
            The result from the update().
            ```
            # Example output
            {
                "project_id": 12345,
                "metadata": [...],
                "media_ids": [123, 124]],
            }
            ```
        """
        project_id = self._client._project_id
        if (
            not media_ids
            or isinstance(media_ids, bool)
            or (not isinstance(media_ids, int) and len(media_ids) == 0)
        ):
            raise ValueError("Missing required flags: {'media_ids'}")

        if not input_metadata:
            raise ValueError("Missing required flags: {'metadata'}")

        dataset_id = self._client.get_project_property(project_id, "dataset_id")

        if isinstance(media_ids, int):
            media_ids = [media_ids]
        else:
            # to avoid errors due to things like numpy.int
            media_ids = list(map(int, media_ids))

        metadata_mapping, id_to_metadata = self._client.get_metadata_mappings(
            project_id
        )

        body = _MetadataUploadRequestBody(
            selectOption=_SelectOption(media_ids),
            project=_Project(project_id, dataset_id),
            metadata=metadata_to_ids(input_metadata, metadata_mapping),
        )

        resp = self._client._api(METADATA_UPDATE, data=obj_to_dict(body))
        resp_data = resp["data"]
        return {
            "project_id": project_id,
            "metadata": ids_to_metadata(resp_data[0]["metadata"], id_to_metadata),
            "media_ids": [media["mediaId"] for media in resp_data],
        }

    def get(self, media_id: int) -> Dict[str, str]:
        """Return all the metadata associated with a given media."""
        resp = self._client._api(
            METADATA_GET, params={"mediaId": media_id, "objectType": "media"}
        )
        _, id_to_metadata = self._client.get_metadata_mappings(self._client._project_id)
        return {id_to_metadata[int(k)]: v for k, v in resp["data"].items()}


class _SelectOption(PrettyPrintable):
    def __init__(self, selected_media: List[int]) -> None:
        self.selected_media = selected_media
        self.unselected_media: List[Union[int, List[int]]] = []
        self.field_filter_map: Dict[str, Any] = {}
        self.column_filter_map: Dict[str, Any] = {}
        self.is_unselect_mode = False


class _Project(PrettyPrintable):
    def __init__(
        self,
        project_id: int,
        dataset_id: int,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id


class _MetadataUploadRequestBody(PrettyPrintable):
    def __init__(
        self,
        selectOption: _SelectOption,
        project: _Project,
        metadata: Dict[str, Any],
    ) -> None:
        self.selectOption = selectOption
        self.project = project
        self.metadata = metadata
