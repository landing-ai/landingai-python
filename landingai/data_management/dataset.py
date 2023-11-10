import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import requests
from PIL import Image
from typing_extensions import deprecated

from tqdm import tqdm

from landingai.common import decode_bitmap_rle
from landingai.data_management.client import (
    _URL_ROOTS,
    GET_FAST_TRAINING_EXPORT,
    GET_PROJECT_MODEL_INFO,
    MEDIA_DETAILS,
    LandingLens,
)
from landingai.data_management.metadata import Metadata
from landingai.exceptions import HttpError

_LOGGER = logging.getLogger(__name__)
_PAGE_SIZE = 10  # 10 is the max page size, unfortunately


class TrainingDataset:
    """A client for fetch the (Fast & East) training dataset."""

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)
        self._metadata_client = Metadata(project_id=project_id, api_key=api_key)

    def get_training_dataset(
        self, output_dir: Path, include_image_metadata: bool = False
    ) -> pd.DataFrame:
        """Get the most recently used training dataset.

        Example output of the returned dataframe:
        ```
                id    split  classes  seg_mask_prediction_path  okng_threshold  media_level_predicted_score    label_id     seg_mask_label_path media_level_label             metadata
        0   11229595   None       []  images/11229595_pred.npy             NaN                          NaN  11301603.0  images/11229595_gt.npy                OK                   {}
        1   11229597   None       []  images/11229597_pred.npy             NaN                          NaN         NaN                    None              None                   {}
        2    9918918  train  [screw]   images/9918918_pred.npy            0.99                     0.954456   8792257.0   images/9918918_gt.npy                NG                   {}
        3    9918924    dev  [screw]   images/9918924_pred.npy            0.99                     0.843393   8792265.0   images/9918924_gt.npy                NG   {'creator': 'bob'}
        4    9918921  train  [screw]   images/9918921_pred.npy            0.99                     0.956114   8792260.0   images/9918921_gt.npy                NG                   {}
        5    9918923  train  [screw]   images/9918923_pred.npy            0.99                     0.943873   8792262.0   images/9918923_gt.npy                NG   {'creator': 'foo'}
        ```

        NOTE:
        1.  Ground truth and prediction masks will be saved to the output_dir as a serialized numpy binary file.
            The file name is the media_id with a suffix of "_gt.npy" or "_pred.npy".
            You can load the numpy array by calling `np.load(file_path)`.
            The shape of the numpy array is (height, width, num_classes).
            The 0th channel is the first class, the 1th channel is the second class and so on. (The background class is not included.)

        2.  For prediction masks, the value of each pixel is the confidence score of the class, i.e. it's not thresholded.
            For ground truth masks, the value of each pixel is either 0 or 1.

        3.  The serialized mask will an empty numpy array when there is no prediction or ground truth mask.
            E.g. the ground truth label is OK, i.e. no defect.
            So be sure to check the shape of the ground truth mask before using it.

        4.  The training dataset could also include images that are not used for training.
            Those images will have a None value for below fields: label_id,seg_mask_label_path,media_level_label
            Tip: for evaluating the model performance, you can filter out those images by checking the label_id field.

        5. The split field could be None, train, dev, or test. None means "unassigned" split.

        6. The metadata field is a dictionary that contains the metadata associated with each image. It's empty by default. Only available when `include_image_metadata` is True.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        project_id = self._client._project_id
        project_model_info = self.get_project_model_info()
        _LOGGER.info(f"Found the most recent model: {project_model_info}")
        model_id = cast(str, project_model_info["registered_model_id"])
        resp = self._client._api(
            route_name=GET_FAST_TRAINING_EXPORT,
            params={
                "projectId": project_id,
                "datasetVersionId": project_model_info["dataset_version_id"],
                "modelId": model_id,
                "skipCreatingDatasetVersion": "true",
            },
        )
        if resp["data"]["project"]["labelType"] != "segmentation":
            raise ValueError(
                f"Project {project_id} is not a segmentation project. Currently only segmentation projects are supported. For other project types, consider using the dataset snapshot export feature from the LandingLens platform UI."
            )

        dataset_id = resp["data"]["dataset"]["id"]
        medias = [
            {
                "id": int(media["media_id"]),
                "split": media.get("split", None),
                "classes": media["defect_list"],
            }
            for media in resp["data"]["data"]
        ]
        _LOGGER.info(
            f"Found {len(medias)} medias in the training dataset. Querying media details..."
        )
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        medias_map = {media["id"]: media for media in medias}
        with tqdm(total=len(medias)) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._get_media_details,
                        media_id=media["id"],
                        dataset_id=dataset_id,
                        model_id=model_id,
                        output_dir=images_dir,
                        include_image_metadata=include_image_metadata,
                    )
                    for media in medias
                ]
                for future in concurrent.futures.as_completed(futures):
                    details = future.result()
                    medias_map[details["id"]].update(details)
                    pbar.update(1)

        return pd.DataFrame(medias)

    def _get_media_details(
        self,
        media_id: int,
        dataset_id: int,
        model_id: str,
        output_dir: Path,
        include_image_metadata: bool,
    ) -> Dict[str, Any]:
        """Get media details and image metadata."""
        resp = self._client._api(
            MEDIA_DETAILS,
            params={
                "mediaId": media_id,
                "datasetId": dataset_id,
                "modelId": model_id,
            },
        )
        data = resp["data"]
        # Get label data
        seg_mask_label_path, label_id, media_level_label = None, None, None
        if data.get("label"):
            label_id = data["label"]["id"]
            media_level_label = data["label"]["mediaLevelLabel"]
            if data["label"].get("annotations") is not None:
                try:
                    flattened_bitmaps = [
                        np.array(
                            decode_bitmap_rle(ann["segmentationBitmapEncoded"]),
                            dtype=np.uint8,
                        ).reshape(
                            (
                                ann["rangeBox"]["ymax"] - ann["rangeBox"]["ymin"] + 1,
                                ann["rangeBox"]["xmax"] - ann["rangeBox"]["xmin"] + 1,
                            )
                        )
                        for ann in data["label"]["annotations"]
                    ]
                    mask = (
                        np.stack(arrays=flattened_bitmaps, axis=2)
                        if flattened_bitmaps
                        else flattened_bitmaps
                    )
                    seg_mask_label_path = output_dir / f"{media_id}_gt.npy"
                    np.save(seg_mask_label_path, mask)
                except Exception:
                    _LOGGER.exception(
                        f"Failed to decode the segmentation mask (prediction) for media {media_id}."
                    )

        # Get prediction data
        seg_mask_prediction_path, media_level_predicted_score, okng_threshold = (
            None,
            None,
            None,
        )
        if data.get("predictionLabel"):
            try:
                okng_threshold = data["predictionLabel"]["okngThreshold"]
                media_level_predicted_score = data["predictionLabel"]["mediaLevelScore"]
                seg_mask_prediction_path = output_dir / f"{media_id}_pred.npy"
                pred_mask = np.asarray(
                    Image.open(
                        requests.get(
                            data["predictionLabel"]["segImgPath"], stream=True
                        ).raw
                    )
                )
                unique_classes = np.unique(pred_mask[:, :, 0])
                masks = []
                for unique_class in unique_classes:
                    assert (
                        unique_class != 0
                    ), "Unexpected data. Background class should not be included in the prediction mask."
                    mask_score = (
                        pred_mask[:, :, 2]
                        * (pred_mask[:, :, 0] == unique_class).astype(np.float16)
                    ) / 255
                    masks.append(mask_score)
                stacked_mask = np.stack(masks, axis=2)
                np.save(seg_mask_prediction_path, stacked_mask)
            except Exception:
                _LOGGER.exception(
                    f"Failed to decode the segmentation mask (label) for media {media_id}."
                )

        media = {
            "id": media_id,
            # prediction data
            "seg_mask_prediction_path": seg_mask_prediction_path.absolute().as_posix()
            if seg_mask_prediction_path
            else None,
            "okng_threshold": okng_threshold,
            "media_level_predicted_score": media_level_predicted_score,
            # label data
            "label_id": label_id,
            "seg_mask_label_path": seg_mask_label_path.absolute().as_posix()
            if seg_mask_label_path
            else None,
            "media_level_label": media_level_label,
        }
        if include_image_metadata:
            metadata = self._metadata_client.get(media_id)
            media["metadata"] = metadata
        return media

    def get_project_model_info(self) -> Dict[str, Union[str, int]]:
        project_id = self._client._project_id
        resp = self._client._api(
            GET_PROJECT_MODEL_INFO, params={"projectId": project_id}
        )
        return {
            "dataset_version_id": resp["data"]["datasetVersionId"],
            "registered_model_id": resp["data"]["registeredModelId"],
        }


@deprecated(
    " You should not use this class unless you're told by the LandingAI team. It's only intended for training flow migration use cases."
)
class LegacyTrainingDataset:
    """A client for fetch the training dataset from legacy training flows."""

    def __init__(self, project_id: int, cookie: str) -> None:
        self._project_id = project_id
        self._cookie = cookie

    def get_legacy_training_dataset(
        self, output_dir: Path, job_id: str
    ) -> pd.DataFrame:
        """Get the training dataset from legacy training flow by job_id.

        Example output of the returned dataframe:
        ```
            media_id   seg_mask_prediction_path         seg_mask_label_path
        0   10413664  /work/landingai-python/104136...  /work/landingai-python/104136...
        1   10413665  /work/landingai-python/104136...  /work/landingai-python/104136...
        2   10413666  /work/landingai-python/104136...  /work/landingai-python/104136...
        ```

        NOTE:
        1. This dataset has a similar format as the dataset returned by `TrainingDataset.get_training_dataset()`.
        2. Only difference is that the prediction mask is thresholded, i.e. the value of each pixel is either 0 or 1.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        data = _fetch_gt_and_predictions(
            self._project_id, self._cookie, job_id=job_id, offset=0
        )
        rows: List[Dict[str, Any]] = [
            _extract_gt_and_predictions(d, output_dir) for d in data["details"]
        ]
        total = data["totalItems"]
        _LOGGER.info(f"Found a total of {total} images:")
        if total > _PAGE_SIZE:
            new_offsets = list(range(0, total - _PAGE_SIZE, _PAGE_SIZE))
            new_offsets = [offset + _PAGE_SIZE for offset in new_offsets]
            with tqdm(total=len(new_offsets)) as pbar:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _fetch_gt_and_predictions,
                            project_id=self._project_id,
                            cookie=self._cookie,
                            job_id=job_id,
                            offset=new_offset,
                        )
                        for new_offset in new_offsets
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        new_data = future.result()
                        new_rows = [
                            _extract_gt_and_predictions(d, output_dir)
                            for d in new_data["details"]
                        ]
                        rows.extend(new_rows)
                        pbar.update(1)
        _LOGGER.info(
            (f"Fetched {len(rows)} image-prediction-label pairs from job {job_id}.")
        )
        return pd.DataFrame(rows)


def _fetch_gt_and_predictions(
    project_id: int, cookie: str, job_id: str, offset: int
) -> Dict[str, Any]:
    root_url = _URL_ROOTS["LANDING_API"]
    url = f"{root_url}/experiment_report/v1/reports/{job_id}/details"
    _LOGGER.info(f"Fetching legacy training dataset predictions from {url}")
    resp = requests.get(
        url=url,
        params={"projectId": project_id, "limit": _PAGE_SIZE, "offset": offset},
        headers={"cookie": cookie},
    )
    if not resp.ok:
        if resp.status_code == 404:
            raise ValueError(
                f"Could not find a classic flow job by job id: {job_id} in project {project_id}. Please double check your job id and project id is correct, and it's a classic flow job."
            )
        error_message = resp.text
        _LOGGER.warning(
            f"Failed to fetch legacy training dataset predictions: {resp.status_code}, {error_message}"
        )
        raise HttpError(
            "HTTP request to LandingLens server failed with "
            f"code {resp.status_code}-{resp.reason} and error message: \n"
            f"{error_message}"
        )
    return cast(Dict[str, Any], resp.json()["data"])


def _extract_gt_and_predictions(
    img_pred_gt_info: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    media_id = int(img_pred_gt_info["mediaId"])
    pred_bitmasks = img_pred_gt_info["prediction"]
    pred_mask_path = _save_mask(pred_bitmasks, output_dir, media_id, save_suffix="pred")
    gt_bitmasks = img_pred_gt_info["groundTruth"]
    gt_mask_path = _save_mask(gt_bitmasks, output_dir, media_id, save_suffix="gt")
    return {
        "media_id": media_id,
        "seg_mask_prediction_path": pred_mask_path.absolute().as_posix(),
        "seg_mask_label_path": gt_mask_path.absolute().as_posix(),
    }


def _save_mask(
    mask_info: Dict[str, Any], output_dir: Path, media_id: int, save_suffix: str
) -> Path:
    h, w = mask_info["imageHeight"], mask_info["imageWidth"]
    if bitmaps := mask_info.get("bitmaps", []):
        index_and_masks = []
        for v in bitmaps.values():
            per_channel_mask = np.array(
                decode_bitmap_rle(v["bitmap"]), dtype=np.uint8
            ).reshape((h, w))
            index_and_masks.append((per_channel_mask, v["labelIndex"]))

        index_and_masks.sort(key=lambda x: x[1])
        masks = [x[0] for x in index_and_masks]
        stacked_mask = np.stack(masks, axis=2)
    else:
        stacked_mask = np.array([])
    mask_path = output_dir / f"{media_id}_{save_suffix}.npy"
    np.save(mask_path, stacked_mask)
    return mask_path
