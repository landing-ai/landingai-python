"""Snowflake-specific adapters and helpers"""

import datetime
from typing import Optional, cast
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
        snowflake_password: str,
        native_app_url: str,
        check_server_ready: bool = True,
    ) -> None:
        super().__init__(
            endpoint_id, api_key=None, check_server_ready=check_server_ready
        )
        self._url = urljoin(native_app_url, "/inference/v1/predict")
        self.snowflake_account = snowflake_account
        self.snowflake_user = snowflake_user
        self.snowflake_password = snowflake_password

        self._auth_token = None
        self._last_auth_token_fetch: Optional[datetime.datetime] = None

    def _load_api_credential(self, api_key: Optional[str]) -> Optional[APIKey]:
        # Snowflake Native App does not use API Key, so we ignore it
        return None

    def _get_auth_token(self) -> str:
        try:
            import snowflake.connector  # type: ignore
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required to use snowflake.NativeAppPredictor. "
                "Please, pip install snowflake-connector-python"
            )

        # Reuse the token if it's not too old
        if self._auth_token is not None and (
            datetime.datetime.now() - self._last_auth_token_fetch
            < self.AUTH_TOKEN_MAX_AGE
        ):
            return self._auth_token

        ctx = snowflake.connector.connect(
            user=self.snowflake_user,
            password=self.snowflake_password,
            account=self.snowflake_account,
            session_parameters={"PYTHON_CONNECTOR_QUERY_RESULT_FORMAT": "json"},
        )
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
        # TODO: Soon, we will remove the need for the apikey header in snowflake native app.
        # if "apikey" in headers:
        #     del headers["apikey"]

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
