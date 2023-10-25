import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import concurrent.futures

import numpy as np
import pandas as pd
import requests
from PIL import Image

from landingai.common import decode_bitmap_rle
from landingai.data_management.client import (
    GET_FAST_TRAINING_EXPORT,
    GET_PROJECT_MODEL_INFO,
    MEDIA_DETAILS,
    LandingLens,
)
from landingai.data_management.metadata import Metadata


_LOGGER = logging.getLogger(__name__)


class TrainingDataset:
    """A client for fetch the (Fast & East) training dataset."""

    def __init__(self, project_id: int, api_key: Optional[str] = None):
        self._client = LandingLens(project_id=project_id, api_key=api_key)
        self._metadata_client = Metadata(project_id=project_id, api_key=api_key)

    def get_training_dataset(
        self, output_dir: Path, include_image_metadata: bool = False
    ) -> pd.DataFrame:
        """Get the most recently used training dataset.

        Ground truth and prediction masks will be saved to the output_dir.
        When the ground truth label is OK, i.e. no defect, the mask will be an empty numpy array.
        So be sure to check the shape of the ground truth mask before using it.

        The training dataset could also include images that are not used for training.
        Those images will have a None value for below fields: label_id,seg_mask_label_path,media_level_label
        Tip: for evaluating the model performance, you can filter out those images by checking the label_id field.
        """
        project_id = self._client._project_id
        project_model_info = self.get_project_model_info()
        model_id = project_model_info["registered_model_id"]
        resp = self._client._api(
            route_name=GET_FAST_TRAINING_EXPORT,
            params={
                "projectId": project_id,
                "datasetVersionId": project_model_info["dataset_version_id"],
                "modelId": model_id,
                "skipCreatingDatasetVersion": "true",
            },
        )
        dataset_id = resp["data"]["dataset"]["id"]
        # class_map = resp["data"]["defectMap"]
        medias = [
            {
                "id": media["media_id"],
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
                shape = (data["properties"]["height"], data["properties"]["width"])
                flattened_bitmaps = [
                    np.array(
                        decode_bitmap_rle(ann["segmentationBitmapEncoded"]), dtype=np.uint8
                    ).reshape(shape)
                    for ann in data["label"]["annotations"]
                ]
                mask = (
                    np.stack(arrays=flattened_bitmaps, axis=2)
                    if flattened_bitmaps
                    else flattened_bitmaps
                )
                seg_mask_label_path = output_dir / f"{media_id}_gt.npy"
                np.save(seg_mask_label_path, mask)

        # Get prediction data
        seg_mask_prediction_path, media_level_predicted_score, okng_threshold = (
            None,
            None,
            None,
        )
        if data.get("predictionLabel"):
            okng_threshold = data["predictionLabel"]["okngThreshold"]
            media_level_predicted_score = data["predictionLabel"]["mediaLevelScore"]
            seg_mask_prediction_path = output_dir / f"{media_id}_pred.png"
            Image.open(
                requests.get(data["predictionLabel"]["segImgPath"], stream=True).raw
            ).save(seg_mask_prediction_path)

        media = {
            "id": media_id,
            # prediction data
            "seg_mask_prediction_path": seg_mask_prediction_path,
            "okng_threshold": okng_threshold,
            "media_level_predicted_score": media_level_predicted_score,
            # label data
            "label_id": label_id,
            "seg_mask_label_path": seg_mask_label_path,
            "media_level_label": media_level_label,
        }
        if include_image_metadata:
            metadata = self._metadata_client.get(media["id"])
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
