import logging
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from landingai.storage.data_access import download_file

_LOGGER = logging.getLogger(__name__)


class SnowflakeCredential(BaseSettings):
    """Snowflake API credential. It's used to connect to Snowflake.
    It supports loading from environment variables or .env files.

    The supported name of the environment variables are (case-insensitive):
    - SNOWFLAKE_USER
    - SNOWFLAKE_PASSWORD
    - SNOWFLAKE_ACCOUNT

    Environment variables will always take priority over values loaded from a dotenv file.
    """

    user: str
    password: str
    account: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SNOWFLAKE_",
        case_sensitive=False,
        extra="ignore",
    )


class SnowflakeDBConfig(BaseSettings):
    """Snowflake connection config.
    It supports loading from environment variables or .env files.

    The supported name of the environment variables are (case-insensitive):
    - SNOWFLAKE_WAREHOUSE
    - SNOWFLAKE_DATABASE
    - SNOWFLAKE_SCHEMA

    Environment variables will always take priority over values loaded from a dotenv file.
    """

    warehouse: str
    database: str
    # NOTE: the name "schema" is reserved by pydantic, so we use "snowflake_schema" instead.
    snowflake_schema: str = Field(..., validation_alias="SNOWFLAKE_SCHEMA")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SNOWFLAKE_",
        case_sensitive=False,
        extra="ignore",
    )


def save_remote_file_to_local(
    remote_filename: str,
    stage_name: str,
    *,
    local_output: Optional[Path] = None,
    credential: Optional[SnowflakeCredential] = None,
    connection_config: Optional[SnowflakeDBConfig] = None,
) -> Path:
    """Save a file stored in Snowflake to local disk.
    If local_output is not provided, a temporary directory will be created and used.
    If credential or connection_config is not provided, it will read from environment variable or .env file instead.
    """
    url = get_snowflake_presigned_url(
        remote_filename,
        stage_name,
        credential=credential,
        connection_config=connection_config,
    )
    if local_output is None:
        local_output = Path(tempfile.mkdtemp())
    file_path = local_output / remote_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, file_output_path=file_path)
    _LOGGER.info(f"Saved file {remote_filename} to {file_path}")
    return file_path


def get_snowflake_presigned_url(
    remote_filename: str,
    stage_name: str,
    *,
    credential: Optional[SnowflakeCredential] = None,
    connection_config: Optional[SnowflakeDBConfig] = None,
) -> str:
    """Get a presigned URL for a file stored in Snowflake.
    NOTE: Snowflake returns a valid URL even if the file doesn't exist.
          So the downstream needs to check if the file exists first.
    """
    import snowflake.connector  # type: ignore

    if credential is None:
        credential = SnowflakeCredential()
    if connection_config is None:
        connection_config = SnowflakeDBConfig()

    ctx = snowflake.connector.connect(
        user=credential.user,
        password=credential.password,
        account=credential.account,
        warehouse=connection_config.warehouse,
        database=connection_config.database,
        schema=connection_config.snowflake_schema,
    )
    cur = ctx.cursor()
    exec_res = cur.execute(f"LIST @{stage_name}")
    if exec_res is None:
        raise ValueError(f"Failed to list files in stage: {stage_name}")
    files = exec_res.fetchall()
    _LOGGER.debug(f"Files in stage {stage_name}: {files}")
    exec_res = cur.execute(
        f"SELECT get_presigned_url(@{stage_name}, '{remote_filename}') as url"
    )
    if exec_res is None:
        raise ValueError(
            f"Failed to get presigned url for file: {remote_filename} in stage: {stage_name}"
        )
    result = exec_res.fetchall()
    if len(result) == 0 or len(result[0]) == 0:
        raise FileNotFoundError(
            f"File ({remote_filename}) not found in stage {stage_name}. Please double check the file exists in the expected location, stage: {stage_name}, db config: {connection_config}."
        )
    result_url: str = result[0][0]
    _LOGGER.info(f"Result url: {result_url}")
    return result_url
