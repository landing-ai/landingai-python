import pytest

# Both cases are cleared because pydantic-settings matches env vars
# case-insensitively, and the tests below set the lowercase spelling.
_API_CREDENTIAL_ENV_VARS = (
    "LANDINGAI_API_KEY",
    "landingai_api_key",
    "LANDINGAI_API_SECRET",
    "landingai_api_secret",
)


@pytest.fixture(autouse=True)
def unset_api_credential_env_vars(monkeypatch):
    """Isolate unit tests from API credentials in the ambient environment.

    CI exports LANDINGAI_API_KEY so the integration tests can call live
    endpoints. Environment variables take precedence over .env files, so
    without this the credential-loading tests would read the real key instead
    of the one they set up, and the tests asserting that an invalid or missing
    key raises would silently find a valid one.

    Tests that need a credential set it themselves after this fixture runs.
    """
    for var in _API_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
