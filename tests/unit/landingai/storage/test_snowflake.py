from unittest import mock
from pathlib import Path

import pytest

from landingai.storage.snowflake import (
    SnowflakeCredential,
    SnowflakeDBConfig,
    get_snowflake_presigned_url,
    save_remote_file_to_local,
)


def test_load_snowflake_settings_from_env_file(tmp_path):
    env_file: Path = tmp_path / ".env"
    env_file.write_text(
        """
        SNOWFLAKE_WAREHOUSE=test_warehouse
        SNOWFLAKE_DATABASE=test_database
        SNOWFLAKE_SCHEMA=test_schema
        """
    )
    # Overwrite the default env_prefix to avoid conflict with the real .env
    SnowflakeDBConfig.model_config["env_file"] = str(env_file)
    snowflake_settings = SnowflakeDBConfig()
    assert snowflake_settings.warehouse == "test_warehouse"
    assert snowflake_settings.database == "test_database"
    assert snowflake_settings.snowflake_schema == "test_schema"
    # reset back to the default config
    SnowflakeDBConfig.model_config["env_file"] = ".env"
    env_file.unlink()


@pytest.mark.skip
@mock.patch("snowflake.connector.connect")
def test_get_snowflake_url(mock_snowflake_connector):
    query_result1 = [
        (
            "s3://landingai-tiger-workspace-dev2/demo_videos/roadway.mp4",
            12783177,
            "f7b7e0c5cb961a0564c0432b0a6213e7",
            "Thu, 18 May 2023 00:29:50 GMT",
        ),
        (
            "s3://landingai-tiger-workspace-dev2/demo_videos/roadway2.mp4",
            12646205,
            "8d70b7f52d72c14ecc492541a17dc15d",
            "Thu, 18 May 2023 00:29:56 GMT",
        ),
    ]
    query_result2 = [
        (
            "https://landingai-tiger-workspace-dev2.s3.us-east-2.amazonaws.com/demo_videos/roadway.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230523T203918Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3599&X-Amz-Credential=AKIA6DXG35RETU7RQM6F%2F20230523%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=f57a9557f1e8c4b44f3dede68d70a084263e00eecb325a7811da1cd82dbf3102",
        )
    ]
    mock_con = mock_snowflake_connector.return_value
    mock_cur = mock_con.cursor.return_value
    mock_cur.execute.return_value.fetchall.side_effect = [query_result1, query_result2]
    credential = SnowflakeCredential(user="test", password="test", account="test")
    connection_config = SnowflakeDBConfig(
        warehouse="XSMALLTEST", database="TIGER_DEMO_DB", snowflake_schema="PUBLIC"
    )
    filename = "roadway.mp4"
    stage_name = "VIDEO_FILES_STAGE"
    url = get_snowflake_presigned_url(
        filename, stage_name, credential=credential, connection_config=connection_config
    )
    assert (
        url
        == "https://landingai-tiger-workspace-dev2.s3.us-east-2.amazonaws.com/demo_videos/roadway.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230523T203918Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3599&X-Amz-Credential=AKIA6DXG35RETU7RQM6F%2F20230523%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=f57a9557f1e8c4b44f3dede68d70a084263e00eecb325a7811da1cd82dbf3102"
    )


@pytest.mark.skip
@pytest.mark.skip(
    reason="This is a real test, which needs to be run manually with a valid snowflake credential."
)
def test_save_remote_file_to_local():
    connection_config = SnowflakeDBConfig(
        warehouse="XSMALLTEST", database="TIGER_DEMO_DB", snowflake_schema="PUBLIC"
    )
    filename = "roadway.mp4"
    stage_name = "VIDEO_FILES_STAGE"
    saved_path = save_remote_file_to_local(
        filename, stage_name, connection_config=connection_config
    )
    print(saved_path)
