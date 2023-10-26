from pathlib import Path

import pytest

from landingai.data_management.dataset import LegacyTrainingDataset, TrainingDataset


@pytest.mark.skip(
    reason="need more work to make it a real test. Test it manually for now."
)
def test_dataset_client__get_dataset_version_id():
    api_key = ""
    dataset = TrainingDataset(project_id=48946424893450, api_key=api_key)
    project_model_info = dataset.get_project_model_info()
    assert project_model_info["dataset_version_id"] == 45815


@pytest.mark.skip(
    reason="need more work to make it a real test. Test it manually for now."
)
def test_dataset_client_get_training_dataset():
    api_key = ""
    dataset = TrainingDataset(project_id=13777903228933, api_key=api_key)
    output = Path("test_fetch_fne_dataset")
    output = dataset.get_training_dataset(output, include_image_metadata=True)
    assert len(output) > 0
    assert output.columns.tolist() == [
        "id",
        "split",
        "classes",
        "seg_mask_prediction_path",
        "okng_threshold",
        "media_level_predicted_score",
        "label_id",
        "seg_mask_label_path",
        "media_level_label",
        "metadata",
    ]


@pytest.mark.skip(
    reason="need more work to make it a real test. Test it manually for now."
)
def test_legacy_dataset_client_get_legacy_training_dataset_predictions():
    cookie = ""
    project_id = 13777903228933
    dataset = LegacyTrainingDataset(project_id=project_id, cookie=cookie)
    # job_id = "cc9576f7-299a-4452-a6f5-6447a2677c80"
    job_id = "3627db83-26be-433a-8406-76c019ebde45"
    output = Path("test_fetch_legacy_dataset")
    output = dataset.get_legacy_training_dataset(output, job_id=job_id)
    print(output)
    assert len(output) > 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
    )
    # test_dataset_client_get_training_dataset()
    test_legacy_dataset_client_get_legacy_training_dataset_predictions()
