import responses

from landingai.data_management.metadata import Metadata

_API_KEY = "land_sk_12345"
_PROJECT_ID = 30863867234314


@responses.activate
def test_set_metadata_for_single_media():
    responses._add_from_file(
        file_path="tests/data/responses/v1_set_metadata_single_media.yaml"
    )
    metadata = Metadata(_PROJECT_ID, _API_KEY)
    resp = metadata.update(10300467, split="train")
    assert resp["project_id"] == _PROJECT_ID
    assert len(resp["media_ids"]) == 1
    response_metadata = resp["metadata"]
    assert len(response_metadata) == 1
    assert response_metadata["split"] == "train"


@responses.activate
def test_set_metadata_for_multiple_media():
    responses._add_from_file(
        file_path="tests/data/responses/v1_set_metadata_multiple_medias.yaml"
    )
    media_ids = [10300467, 10300466, 10300465]
    metadata = Metadata(_PROJECT_ID, _API_KEY)
    resp = metadata.update(media_ids, creator="tom")
    assert resp["project_id"] == _PROJECT_ID
    assert resp["media_ids"] == media_ids
    response_metadata = resp["metadata"]
    assert len(response_metadata) == 1
    assert response_metadata["creator"] == "tom"


def test_get_metadata_by_media_id():
    _API_KEY = "land_sk_aMemWbpd41yXnQ0tXvZMh59ISgRuKNRKjJEIUHnkiH32NBJAwf"
    metadata_client = Metadata(_PROJECT_ID, api_key=_API_KEY)
    result = metadata_client.get(10304143)
    assert result["creator"] == "123"
