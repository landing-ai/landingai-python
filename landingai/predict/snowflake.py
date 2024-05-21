"""Snowflake-specific adapters and helpers"""

import datetime
from typing import Any, Dict, Optional, cast
from urllib.parse import urljoin

from requests import Session
from landingai.common import APIKey
from landingai.predict.cloud import Predictor
from landingai.predict.utils import create_requests_session


class SnowflakeNativeAppPredictor(Predictor):
    """Snowflake Native App Predictor, which is basically a regular cloud predictor with a different auth mechanism."""

    # For how long can we reuse the auth token before having to fetch a new one
    AUTH_TOKEN_MAX_AGE = datetime.timedelta(minutes=5)

    def __init__(
        self,
        endpoint_id: str,
        *,
        snowflake_account: str,
        snowflake_user: str,
        snowflake_password: Optional[str] = None,
        snowflake_private_key: Optional[str] = None,
        native_app_url: str,
        # TODO: Remove this once we remove the API key auth from snowflake
        api_key: Optional[str] = None,
        check_server_ready: bool = True,
    ) -> None:
        assert (
            snowflake_password is not None or snowflake_private_key is not None
        ), "You must provide either `snowflake_password` or `snowflake_public_key`."
        super().__init__(
            endpoint_id, api_key=api_key, check_server_ready=check_server_ready
        )
        self._url = urljoin(native_app_url, "/inference/v1/predict")
        self.snowflake_account = snowflake_account
        self.snowflake_user = snowflake_user
        self.snowflake_password = snowflake_password
        self.snowflake_private_key = snowflake_private_key

        self._auth_token = None
        self._last_auth_token_fetch: Optional[datetime.datetime] = None

    def _load_api_credential(self, api_key: Optional[str]) -> Optional[APIKey]:
        # Snowflake Native App does not use API Key, so we ignore it.
        # Once we remove the API key auth from snowflake, we can always return None here.
        if api_key is None:
            return None
        return super()._load_api_credential(api_key)

    def _get_auth_token(self) -> str:
        try:
            import snowflake.connector  # type: ignore
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError(
                "In order to use snowflake.NativeAppPredictor, you must install snowflake optionals. "
                "Please, run: pip install landingai[snowflake]"
            )

        # Reuse the token if it's not too old
        if self._auth_token is not None and (
            datetime.datetime.now() - self._last_auth_token_fetch
            < self.AUTH_TOKEN_MAX_AGE
        ):
            return self._auth_token
        connect_params: Dict[str, Any] = dict(
            user=self.snowflake_user,
            account=self.snowflake_account,
            session_parameters={"PYTHON_CONNECTOR_QUERY_RESULT_FORMAT": "json"},
        )
        if self.snowflake_password is not None:
            connect_params["password"] = self.snowflake_password
        if self.snowflake_private_key is not None:
            p_key = serialization.load_pem_private_key(
                self.snowflake_private_key.encode("ascii"),
                password=None,
                backend=default_backend(),
            )
            connect_params["private_key"] = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

        ctx = snowflake.connector.connect(**connect_params)
        ctx._all_async_queries_finished = lambda: False  # type: ignore
        token_data = ctx._rest._token_request("ISSUE")  # type: ignore
        self._auth_token = token_data["data"]["sessionToken"]
        self._last_auth_token_fetch = datetime.datetime.now()
        return cast(str, self._auth_token)

    @property
    def _session(self) -> Session:
        extra_x_event = {
            "endpoint_id": self._endpoint_id,
            "model_type": "fast_and_easy",
        }
        headers = self._build_default_headers(self._api_credential, extra_x_event)
        headers["Authorization"] = f'Snowflake Token="{self._get_auth_token()}"'

        return create_requests_session(
            url=self._url,
            num_retry=self._num_retry,
            headers=headers,
        )

    @_session.setter
    def _session(self, value: Session) -> None:
        """Ignore setting the session. We always create a new session when needed."""
        pass
