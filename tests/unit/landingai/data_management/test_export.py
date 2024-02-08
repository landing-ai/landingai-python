import responses

from landingai.data_management.export import Exporter


@responses.activate
def test_export_event_logs():
    responses._add_from_file(file_path="tests/data/responses/test_export_event_logs.yaml")
    project_id = 12345
    api_key = "land_sk_12345"
    client = Exporter(project_id, api_key)
    res = client.export_event_logs("2021-06-01 00:00:00.000", "2024-01-30 00:00:00.000")
    assert res == {
        "s3Path": "s3://.../1622505600-1706572800.csv",
        "signedUrl": "https://landinglens-bucket.s3.../1622505600-1706572800.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential...-SignedHeaders=host"
    }
