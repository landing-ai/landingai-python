import responses

from landingai.data_management.export import Exporter
from freezegun import freeze_time
from unittest import mock


@responses.activate
@mock.patch("landingai.data_management.export.Exporter._download_file_from_signed_url")
@freeze_time("2024-01-30 00:00:00.000000")  # Mock the current date and time
def test_export_event_logs(mocked_method):
    responses._add_from_file(
        file_path="tests/data/responses/test_export_event_logs.yaml"
    )
    project_id = 12345
    api_key = "land_sk_12345"
    client = Exporter(project_id, api_key)
    res = client.export_event_logs("2021-06-01", "./local/path/to/save/file.csv")
    assert res is None
