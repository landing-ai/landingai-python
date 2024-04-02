from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import responses
from PIL import Image

from landingai.data_management.media import Media
from landingai.exceptions import HttpError

_API_KEY = "123"
_PROJECT_ID = 30863867234314


"""
Tests for Media.update_split_key
"""


@responses.activate
def test_update_split_key_bulk_update(caplog):
    responses._add_from_file(
        file_path="tests/data/responses/test_update_split_key_bulk_update.yaml"
    )
    media = Media(21529989074947, _API_KEY)
    res = media.ls()
    media_ids = [m["id"] for m in res["medias"]]
    media_ids_to_update = media_ids[:2]
    media.update_split_key(media_ids_to_update, "dev")
    assert (
        "Successfully updated split key to 'dev' for 2 medias with media ids: [10607910, 10607881]"
        in caplog.text
    )
    media.update_split_key([media_ids_to_update[0]], "train")
    assert (
        "Successfully updated split key to 'train' for 1 medias with media ids: [10607910]"
        in caplog.text
    )


@responses.activate
def test_update_split_key_unassigned(caplog):
    responses._add_from_file(
        file_path="tests/data/responses/test_update_split_key_unassigned.yaml"
    )
    media = Media(21529989074947, _API_KEY)
    res = media.ls()
    media_ids = [m["id"] for m in res["medias"]]
    media_ids_to_update = media_ids[2:4]
    media.update_split_key(media_ids_to_update, "")
    assert (
        "Successfully updated split key to '' for 2 medias with media ids: [10604095, 10604010]"
        in caplog.text
    )


"""
Tests for Media.upload
"""


@responses.activate
def test_single_file_upload(tmp_path):
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_upload_single_file.yaml"
    )
    img_name = "image_1688427395107.jpeg"

    media = Media(_PROJECT_ID, _API_KEY)
    file_name, img_path = _write_random_test_image(tmp_path, file_name=img_name)

    resp = media.upload(img_path)
    assert resp["num_uploaded"] == 1
    assert resp["skipped_count"] == 0
    assert resp["error_count"] == 0
    assert len(resp["medias"]) == 1
    assert file_name in resp["medias"][0]["path"]
    assert resp["medias"][0]["id"] is not None
    assert resp["medias"][0]["uploadTime"] is not None
    assert resp["medias"][0]["properties"] == {
        "width": 20,
        "format": "jpeg",
        "height": 20,
        "orientation": 1,
    }
    resp = media.ls()
    medias = resp["medias"]
    assert len(medias) >= 1
    assert file_name == medias[0]["name"]


@mock.patch("landingai.data_management.media.LandingLens")
def test_single_file_upload_400(mocked_ll_client, tmp_path):
    ll_client_instance = mocked_ll_client.return_value
    ll_client_instance._api_async.side_effect = HttpError(
        "HTTP request to LandingLens server failed with code 400-Bad Request and error message: \nInvalid xml file format"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    file_name, img_path = _write_random_test_image(
        tmp_path, file_name="image_1688427395107.jpeg"
    )
    resp = media.upload(img_path)
    assert resp["num_uploaded"] == 0
    assert resp["skipped_count"] == 0
    assert resp["error_count"] == 1
    assert len(resp["medias"]) == 0
    assert (
        resp["files_with_errors"][file_name]
        == "HTTP request to LandingLens server failed with code 400-Bad Request and error message: \nInvalid xml file format"
    )


@responses.activate
def test_single_file_upload_metadata(tmp_path):
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_upload_metadata.yaml"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    file_name, img_path = _write_random_test_image(tmp_path, file_name="image.jpeg")
    resp = media.upload(img_path, metadata_dict={"test": "test"})
    assert resp["num_uploaded"] == 1
    assert resp["skipped_count"] == 0
    assert resp["error_count"] == 0
    assert len(resp["medias"]) == 1
    assert file_name in resp["medias"][0]["path"]
    assert resp["medias"][0]["id"] is not None
    assert resp["medias"][0]["uploadTime"] is not None
    assert resp["medias"][0]["properties"] == {
        "width": 20,
        "format": "jpeg",
        "height": 20,
        "orientation": 1,
    }
    resp = media.ls()
    medias = resp["medias"]
    assert len(medias) >= 1
    assert file_name == medias[0]["name"]


@responses.activate
def test_folder_upload_with_subdirectories_skip_txt(tmp_path):
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_upload_folder.yaml"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    _write_random_test_image_folder(tmp_path, num_images=1)
    _write_random_test_image_folder(tmp_path / "subdir", num_images=2)
    upload_resp = media.upload(tmp_path)
    assert upload_resp["skipped_count"] == 2
    assert upload_resp["num_uploaded"] == 3
    assert upload_resp["error_count"] == 0


def _write_random_test_image(folder_path, file_name=None, image_name_prefix="image"):
    if not file_name:
        file_name = f"{image_name_prefix}_{int(datetime.now().timestamp() * 1000)}.jpeg"
    img_path = folder_path / file_name
    arr = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return file_name, img_path


def _write_random_test_image_folder(folder_path, num_images=2):
    folder_path.mkdir(parents=True, exist_ok=True)
    (folder_path / "irrelevant.txt").touch()
    files = []
    for _, i in enumerate(range(num_images)):
        files.append(
            _write_random_test_image(folder_path, image_name_prefix=f"image_{i}")
        )
    return files


"""
Tests for Media.ls
"""


@responses.activate
def test_ls_filter_by_split():
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_list_filter_by_split.yaml"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    resp = media.ls(split="dev")
    medias = resp["medias"]
    assert len(medias) == 2


@responses.activate
def test_ls_media_statuses_filter_by_one_status():
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_list_by_one_status.yaml"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    output = media.ls(media_status="raw")
    assert len(output["medias"]) >= 10


@responses.activate
def test_ls_media_statuses_mixed_ordering():
    responses._add_from_file(
        file_path="tests/data/responses/v1_media_list_by_three_status.yaml"
    )
    media = Media(_PROJECT_ID, _API_KEY)
    output = media.ls(media_status=["raw", "approved", "in_task"])
    assert len(output["medias"]) >= 10


def test_ls_pagination_step_exceeds_max():
    media = Media(_PROJECT_ID, _API_KEY)
    media._media_max_page_size = 2
    media._metadata_max_page_size = 1
    with pytest.raises(ValueError) as e:
        media.ls()
    assert str(e.value) == "Exceeded max page size of 2"


def test_ls_not_allowed_media_status():
    media = Media(_PROJECT_ID, _API_KEY)
    with pytest.raises(ValueError) as e:
        media.ls(media_status="sth_wrong")
    assert (
        str(e.value)
        in "Wrong media status. Allowed media statuses are ['raw', 'in_task', 'approved']"
    )


def test_ls_not_allowed_media_status_list_with_one_wrong():
    media = Media(_PROJECT_ID, _API_KEY)
    with pytest.raises(ValueError) as e:
        media.ls(media_status=["raw", "pending_sth_wrong"])
    assert (
        str(e.value)
        in "Wrong media status. Allowed media statuses are ['raw', 'in_task', 'approved']"
    )


def test_ls_not_allowed_media_status_empty_list():
    media = Media(_PROJECT_ID, _API_KEY)
    with pytest.raises(ValueError) as e:
        media.ls(media_status=[])
    assert (
        str(e.value)
        in "Wrong media status. Allowed media statuses are ['raw', 'in_task', 'approved']"
    )
