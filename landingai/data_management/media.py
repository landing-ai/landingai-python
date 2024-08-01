import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast


import PIL.Image
from PIL.Image import Image
from tqdm import tqdm

from landingai.data_management.client import (
    GET_PROJECT_SPLIT,
    MEDIA_LIST,
    MEDIA_UPDATE_SPLIT,
    MEDIA_UPLOAD,
    LandingLens,
)
from landingai.data_management.utils import (
    PrettyPrintable,
    obj_to_params,
    validate_metadata,
    metadata_to_ids,
)
from landingai.exceptions import DuplicateUploadError, HttpError
from landingai.utils import _LLENS_SUPPORTED_IMAGE_FORMATS


MediaType = Enum("MediaType", ["image", "video"])
SrcType = Enum("SrcType", ["user", "production_line", "prism"])
ThumbnailSize = Enum(
    "ThumbnailSize", ["50x50", "250x250", "500x500", "750x750", "1000x1000"]
)


_ALLOWED_EXTENSIONS = _LLENS_SUPPORTED_IMAGE_FORMATS + ["TIFF", "TIF"]
_HIDDEN_FILES_TO_IGNORE = ["thumbs.db", "desktop.ini", ".ds_store"]
_SUPPORTED_KEYS = {"train", "dev", "test", ""}
_CONCURRENCY_LIMIT = 5
_LOGGER = logging.getLogger(__name__)


class Media:
    """Media management API client.
    This class provides a set of APIs to manage the medias (images) uploaded to LandingLens.
    For example, you can use this class to upload medias (images) to LandingLens or list
    the medias are already uploaded to the LandingLens.

    Example
    -------
    >>> client = Media(project_id, api_key)
    >>> client.upload("path/to/image.jpg")
    >>> client.upload("path/to/image_folder")
    >>> print(client.ls())

    Parameters
    ----------
    project_id: int
        LandingLens project id.  Can override this default in individual commands.
    api_key: Optional[str]
        LandingLens API Key. If it's not provided, it will be read from the environment
        variable LANDINGAI_API_KEY, or from .env file on your project root directory.
    """

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)
        self._media_max_page_size = 1000
        self._metadata_max_page_size = 500

    def upload(
        self,
        source: Union[str, Path, Image],
        split: str = "",
        classification_name: Optional[str] = None,
        object_detection_xml: Optional[str] = None,
        seg_mask: Optional[str] = None,
        seg_defect_map: Optional[str] = None,
        nothing_to_label: bool = False,
        metadata_dict: Optional[Dict[str, Any]] = None,
        validate_extensions: bool = True,
        tolerate_duplicate_upload: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload media to platform.

        Parameters
        ----------
        source: Union[str, Path, Image]
            The image source to upload. It can be a path to the local image file, an
            image folder or a PIL Image object. For image files, the supported formats
            are jpg, jpeg, png, bmp and tiff.
        split: str
            Set this media to one split ('train'/'dev'/'test'), '' represents Unassigned
            and is the default
        classification_name: str
            Set the media's classification if the project type is Classification or
            Anomaly Detection
        object_detection_xml: str
            Path to the Pascal VOC xml file for object detection project
        seg_mask: str
            Path to the segmentation mask file for segmentation project
        seg_defect_map: str
            Path to the segmentation defect_map.json file for segmentation project.
            To get this map, you can use the `landingai.data_management.label.Label` API.
            See below code as an example.
            ```python
            >>> client = Label(project_id, api_key)
            >>> client.get_label_map()
            >>> {'0': 'ok', '1': 'cat', '2': 'dog'}
            ```
        nothing_to_label: bool
            Set the media's label as OK, valid for object detection and segmetation
            project
        metadata_dict: dict
            A dictionary of metadata to be updated or inserted. The key of the metadata
            needs to be created/registered (for the first time) on LandingLens before
            media uploading.
        validate_extensions: bool
            Defaults to True. Files other than jpg/jpeg/png/bmp will be skipped.
            If set to False, will try to upload all files. Behavior of platform
            for unexpected extensions may not be correct - for example, most likely file
            will be uploaded to s3, but won't show in data browser.
        tolerate_duplicate_upload: bool
            Whether to tolerate duplicate upload. A duplicate upload is identified by
            status code 409. The server returns a 409 status code if the same media file
            content exists in the project. Defaults to True. If set to False, will raise
            a `landingai.exceptions.HttpError` if it's a duplicate upload.

        Returns
        -------
        Dict[str, Any]
            The result from the upload().
            ```
            # Example output
            {
                "num_uploaded": 10,
                "skipped_count": 0,
                "error_count": 0,
                "medias": [...],
                "files_with_errors": {},
            }
            ```
        """
        if isinstance(source, Path):
            source = str(source)
        if isinstance(source, str) and not os.path.exists(source):
            raise ValueError(
                f"file/folder does not exist at the specified path {source}"
            )

        project_id = self._client._project_id
        project = self._client.get_project_property(project_id)
        dataset_id = project.get("datasetId")
        label_type = project.get("labelType")

        # construct initial_label
        initial_label: Dict[str, Any] = {}
        if nothing_to_label:
            initial_label["unlabeledAsNothingToLabel"] = True
        elif (
            label_type == "classification" or label_type == "anomaly_detection"
        ) and classification_name is not None:
            initial_label["classification"] = classification_name
        elif label_type == "bounding_box" and object_detection_xml is not None:
            xml_content = open(object_detection_xml, "rb").read()
            initial_label["objectDetection"] = base64.b64encode(xml_content).decode(
                "utf-8"
            )
        elif (
            label_type == "segmentation"
            and seg_mask is not None
            and seg_defect_map is not None
        ):
            seg_defect_map_content = open(seg_defect_map, "r").read()
            seg_mask_content = open(seg_mask, "rb").read()
            initial_label["segMask"] = base64.b64encode(seg_mask_content).decode(
                "utf-8"
            )
            initial_label["segDefectMap"] = seg_defect_map_content

        # construct metadata
        metadata: Dict[str, Any] = {} if metadata_dict is None else metadata_dict
        if metadata != {}:
            metadata_mapping, _ = self._client.get_metadata_mappings(project_id)
            metadata = metadata_to_ids(metadata, metadata_mapping)

        medias: List[Dict[str, Any]] = []
        skipped_count = 0
        error_count = 0
        medias_with_errors: Dict[str, Any] = {}

        assert isinstance(source, (str, Image))
        if isinstance(source, str) and os.path.isdir(source):
            (
                medias,
                skipped_count,
                error_count,
                medias_with_errors,
            ) = _upload_folder(
                self._client,
                dataset_id,
                source,
                project_id,
                validate_extensions,
                tolerate_duplicate_upload,
            )
        else:
            # Resolve filename and extension for _upload_media()
            if isinstance(source, Image):
                ext = "png"
                ts = int(datetime.now().timestamp() * 1000)
                filename = f"image_{ts}.{ext}"
            else:
                assert isinstance(source, str)
                filename = os.path.basename(source)
                ext = os.path.splitext(filename)[-1][1:]
            # Validate extension
            if validate_extensions and ext.upper() not in _ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"""Unexpected extension {ext}. Allowed extensions are: {_ALLOWED_EXTENSIONS}.
                    If you want to attempt the upload anyway, set validate_extensions=False.
                    This may result in an unexpected behavior - e.g. file not showing up in data browser."""
                )
            try:
                resp = _upload_media(
                    self._client,
                    dataset_id,
                    filename,
                    source,
                    project_id,
                    ext,
                    split,
                    initial_label,
                    metadata,
                    tags,
                )
                medias.append(resp)
            except DuplicateUploadError:
                if not tolerate_duplicate_upload:
                    raise
                skipped_count = 1
            except Exception as e:
                error_count = 1
                medias_with_errors[filename] = str(e)

        return {
            "num_uploaded": len(medias),
            "skipped_count": skipped_count,
            "error_count": error_count,
            "medias": medias,
            "files_with_errors": medias_with_errors,
        }

    def ls(
        self,
        offset: int = 0,
        limit: int = 1000,
        media_status: Union[str, List[str], None] = None,
        **metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        List medias with metadata for given project id. Can be filtered using metadata.
        NOTE: pagination is applied with the `offset` and `limit` parameters.

        Parameters
        ----------
        offset: int
            Defaults to 0. As in standard pagination.
        limit: int
            Max 1000. Defaults to 1000. As in standard pagination.
        media_status: Union[str, List]
            Gets only medias with specified statuses. Defaults to None - then medias
            with all statuses are fetched.
            Possible values: raw, pending_labeling, pending_review, rejected, approved
        **metadata:
            Kwargs used as metadata that will be used for server side filtering of the results.
        """
        if limit - offset > self._media_max_page_size:
            raise ValueError(f"Exceeded max page size of {self._media_max_page_size}")

        if media_status is not None:
            _validate_media_status(media_status)

        project_id = self._client._project_id
        assert project_id is not None

        dataset_id = self._client.get_project_property(project_id, "dataset_id")

        metadata_mapping, meta_id_to_metadata = self._client.get_metadata_mappings(
            project_id
        )
        metadata_filter_map: Dict[str, Any] = {}
        if metadata and len(metadata) > 0:
            metadata_filter_map = _metadata_to_filter(metadata, metadata_mapping)

        column_filter_map: Dict[str, Any] = {}
        if media_status is not None:
            if isinstance(media_status, str):
                media_status = [media_status]
            column_filter_map = {
                "datasetContent": {"mediaStatus": {"CONTAINS_ANY": media_status}}
            }

        resp = self._client._api(
            MEDIA_LIST,
            params=_build_list_media_request(
                limit,
                column_filter_map,
                dataset_id,
                metadata_filter_map,
                offset,
                project_id,
            ),
        )
        medias = resp["data"]
        # convert the metadata ids to metadata names
        for media in medias:
            media["metadata"] = {
                meta_id_to_metadata.get(int(k), None): v
                for k, v in media["metadata"].items()
            }

        if len(medias) == self._media_max_page_size:
            _LOGGER.warning(f"fetched medias only up to {self._media_max_page_size}")

        return {
            "medias": medias,
            "num_requested": limit - offset,
            "count": len(medias),
            "offset": offset,
            "limit": limit,
        }

    def update_split_key(
        self,
        media_ids: List[int],
        split_key: str,
    ) -> None:
        """
        Update the split key for a list of medias on the LandingLens platform.

        Parameters
        ----------
        media_ids: List[int]
            A list of media ids to update split key.
        split: str
            The split key to set for these medias, it could be 'train', 'dev', 'test' or '' (where '' represents Unassigned) and is the default.

        Example
        -------
        >>> client = Media(project_id, api_key)
        >>> client.update_split_key(media_ids=[1001, 1002], split_key="test")  # assign split key 'test' for media ids 1001 and 1002
        >>> client.update_split_key(media_ids=[1001, 1002], split_key="")    # remove split key for media ids 1001 and 1002

        """
        split_key = split_key.strip().lower()
        if split_key not in _SUPPORTED_KEYS:
            raise ValueError(
                f"Invalid split key: {split_key}. Supported split keys are: {_SUPPORTED_KEYS}"
            )
        project_id = self._client._project_id
        split_id = 0  # 0 is Unassigned split
        if split_key != "":
            resp = self._client._api(
                GET_PROJECT_SPLIT, params={"projectId": project_id}
            )
            split_name_to_id = {
                split["splitSetName"].lower(): split["id"] for split in resp["data"]
            }
            assert (
                split_key in split_name_to_id
            ), f"Split key {split_key} not found in project {project_id}. Available split keys in this project are: {split_name_to_id.keys()}"
            split_id = split_name_to_id[split_key]
        dataset_id = self._client.get_project_property(project_id)["datasetId"]
        self._client._api(
            MEDIA_UPDATE_SPLIT,
            params={
                "projectId": project_id,
                "datasetId": dataset_id,
                "splitSet": split_id,
                "selectMediaOptions": json.dumps({"selectedMedia": media_ids}),
            },
        )
        _LOGGER.info(
            f"Successfully updated split key to '{split_key}' for {len(media_ids)} medias with media ids: {media_ids}"
        )


class _SortOptions(PrettyPrintable):
    def __init__(self, offset: int, limit: int) -> None:
        self.offset = offset
        self.limit = limit


class _ListMediaRequestParams(PrettyPrintable):
    def __init__(
        self,
        project_id: int,
        dataset_id: int,
        sort_options: _SortOptions,
        column_filter_map: Optional[Dict[str, Any]] = None,
        metadata_filter_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.sort_options = sort_options
        if column_filter_map is None:
            column_filter_map = {}
        self.column_filter_map = column_filter_map
        if metadata_filter_map is None:
            metadata_filter_map = {}
        self.metadata_filter_map = metadata_filter_map
        self.columns_to_return = [
            "id",
            "mediaType",
            "srcType",
            "srcName",
            "properties",
            "name",
            "uploadTime",
            "mediaStatus",
            "metadata",
        ]


def _metadata_to_filter(
    input_metadata: Dict[str, Any],
    metadata_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    validate_metadata(input_metadata, metadata_mapping)
    return {
        metadata_mapping[key][0]: {
            "CONTAINS_ANY": [val] if not isinstance(val, list) else val
        }
        for key, val in input_metadata.items()
        if key in metadata_mapping
    }


def _validate_media_status(media_status: Union[str, List[str]]) -> None:
    allowed_media_statuses = [
        "raw",
        "in_task",
        "approved",
    ]
    if not (isinstance(media_status, str) or isinstance(media_status, list)):
        raise ValueError("Media status must be a string or a list")
    if (
        isinstance(media_status, str) and media_status not in allowed_media_statuses
    ) or (
        isinstance(media_status, list)
        and (
            len(media_status) == 0
            or len(set(media_status) - set(allowed_media_statuses)) > 0
        )
    ):
        raise ValueError(
            f"Wrong media status. Allowed media statuses are {allowed_media_statuses}"
        )


def _build_list_media_request(
    limit: int,
    column_filter_map: Dict[str, Any],
    dataset_id: int,
    metadata_filter_map: Dict[str, Any],
    offset: int,
    project_id: int,
) -> Dict[str, Any]:
    return obj_to_params(
        _ListMediaRequestParams(
            project_id=project_id,
            dataset_id=dataset_id,
            # server has limit per call
            sort_options=_SortOptions(offset=offset, limit=limit),
            column_filter_map=column_filter_map,
            metadata_filter_map=metadata_filter_map,
        )
    )


def _upload_folder(
    client: LandingLens,
    dataset_id: int,
    path: str,
    project_id: int,
    validate_extensions: bool,
    tolerate_duplicate_upload: bool,
) -> Tuple[List[Any], int, int, Dict[str, Any]]:
    error_count = 0
    skipped_count = 0
    medias = []
    medias_with_errors = {}
    thread_pool = ThreadPoolExecutor(max_workers=_CONCURRENCY_LIMIT)
    tasks = []

    for root, _, filenames in os.walk(path):
        _LOGGER.info(f"Uploading {len(filenames)} files")
        for filename in tqdm(filenames):
            ext = os.path.splitext(filename)[-1][1:]
            if filename.lower() in _HIDDEN_FILES_TO_IGNORE:
                pass
            elif ext.upper() in _ALLOWED_EXTENSIONS or not validate_extensions:
                task = thread_pool.submit(
                    _upload_media,
                    client,
                    dataset_id,
                    filename,
                    os.path.join(root, filename),
                    project_id,
                    ext,
                )
                tasks.append(task)
            else:
                skipped_count += 1
    for task in tqdm(as_completed(tasks), total=len(tasks)):
        try:
            result = task.result()
            medias.append(result)
        except DuplicateUploadError:
            if not tolerate_duplicate_upload:
                raise
            skipped_count += 1
        except Exception as e:
            error_count += 1
            medias_with_errors[filename] = e

    return medias, skipped_count, error_count, medias_with_errors


def _upload_media(
    client: LandingLens,
    dataset_id: int,
    filename: str,
    source: Union[str, Image],
    project_id: int,
    ext: str,
    split: str = "",
    initial_label: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # The platform doesn't support tiff, so we convert it to png before uploading
    if ext.lower() == "tiff":
        ext = "png"
        filename = filename.replace(".tiff", ".png")
        assert isinstance(source, str)
        source = PIL.Image.open(source)  # later it gets converted to png in bytes

    filetype = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    form_data = {
        "project_id": str(project_id),
        "dataset_id": str(dataset_id),
        "name": filename,
        "file": (filename, open(source, "rb"), "text/plain")
        if isinstance(source, str)
        else source,
        "split": split,
        "initial_label": json.dumps(initial_label),
        "tags": json.dumps(tags),
        "metadata": json.dumps(metadata),
    }
    resp = client._api_async(MEDIA_UPLOAD, form_data=form_data)
    if resp["code"] == 409:
        _LOGGER.debug(
            f"Skipping Media ({filename}, {filetype}) as it already exists in project ({project_id}), response: {resp}"
        )
        raise DuplicateUploadError(
            f"Skipping Media ({filename}, {filetype}) as it already exists in project ({project_id}), response: {resp}"
        )
    elif "data" not in resp:
        raise HttpError(
            f"Failed to upload media due to HTTP {resp['code']} error, reason: {resp['message']}"
        )

    return cast(Dict[str, Any], resp["data"])
