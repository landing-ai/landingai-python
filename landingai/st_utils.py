"""This module contains common streamlit utilities that are used across the example Apps in this repo.
It's only intended for the example streamlit Apps. When using the SDK in your own project, you don't need to use this module.
"""
import logging
import os
from typing import Any, Optional

_DEFAULT_API_KEY_ENV_VAR = "LANDINGAI_API_KEY"

_LOGGER = logging.getLogger(__name__)


def is_running_in_streamlit() -> bool:
    """Return True if the code is running in a streamlit App."""
    # See: https://discuss.streamlit.io/t/how-to-check-if-code-is-run-inside-streamlit-and-not-e-g-ipython/23439/2
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


def _import_st() -> Any:
    """Import streamlit and raise an error if it fails."""
    try:
        import streamlit as st

        return st
    except ImportError as e:
        raise ValueError(
            """Failed to import streamlit due to missing the `streamlit` dependency.
This is likely because you are trying to use a function in this module outside of the example Apps in this repo. But this function is only intended for example Apps of this repo.
If you are using this function in other environment, please install streamlit manually first, e.g. 'pip install streamlit'.
If you are running one of the example Apps of this repo, please follow the installation instructions of that App.
    """
        ) from e


def setup_page(page_title: str) -> None:
    """Common setup code for streamlit pages.
    This function should be called only once at the beginning of the page.
    Commoon setup includes:
    1. Set page title and favicon
    2. Set up logging configuration
    3. Hide the default streamlit menu button and footer
    """
    level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s][%(filename)s] %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    st = _import_st()
    st.set_page_config(
        page_title=page_title,
        page_icon="./examples/apps/assets/favicon.ico",
        layout="centered",
        initial_sidebar_state="auto",
    )
    # Hide the default menu button (on the top right corner) and the footer
    hide_footer_style = """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        #MainMenu {
                visibility: hidden;
            }

        footer {
                visibility: hidden;
            }
        </style>
        """
    st.markdown(hide_footer_style, unsafe_allow_html=True)


def render_svg(svg: str, margin_bottom: int = 3) -> None:
    """Renders the given svg string.
    This is a workaround as st.image doesn't seem to work.
    See: https://github.com/streamlit/streamlit/issues/275
    """
    st = _import_st()
    import base64

    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"  height="50" />' % b64
    st.write(html, unsafe_allow_html=True)
    for _ in range(margin_bottom):
        st.write("\n")


def check_api_credentials_set() -> None:
    """Check if API credential is set in session state."""
    st = _import_st()
    api_key = st.session_state.get("api_key")
    if api_key:
        return

    st.error("Please open the sidebar and enter your API key first.")
    st.stop()


def check_endpoint_id_set() -> None:
    """Check if endpoint ID is set in session state."""
    st = _import_st()
    if st.session_state.get("endpoint_id"):
        return
    st.error("Please open the sidebar and enter your CloudInference endpoint ID first.")
    st.stop()


def get_api_key_or_use_default() -> Optional[str]:
    """Get API key (v2) from the session state.
    If the API key is not set in the session state, it will look for the default API key from environment variables.
    """
    st = _import_st()
    key: str = st.session_state["api_key"]
    if not key:
        return get_default_api_key()
    return key


def get_default_api_key() -> Optional[str]:
    """Get the default (free trial) API key."""
    default_key = os.environ.get(_DEFAULT_API_KEY_ENV_VAR)
    if not default_key:
        _LOGGER.warning(
            "The default API key is not set in the application. Please set your API key in the environment variable. E.g. export LANDINGAI_API_KEY=......"
        )
        return None
    return default_key


def render_api_config_form(
    render_endpoint_id: bool = False,
    default_key: str = "",
    default_endpoint_id: str = "",
) -> None:
    """Show API credential and endpoint ID (optionally) configuration form.
    The form provides a way for users to set the API credential and endpoint ID in the session state.
    The values saved in the session state can be accessed by `get_api_credential_or_use_default()`.
    This is the source of truth for the API credential and endpoint ID in the App.
    """

    st = _import_st()
    # Set a default value in session state for the first time
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = default_key
    if "endpoint_id" not in st.session_state:
        st.session_state["endpoint_id"] = default_endpoint_id

    def _update_credential(api_key: str, endpoint_id: Optional[str]) -> None:
        st.session_state["api_key"] = api_key
        if endpoint_id:
            st.session_state["endpoint_id"] = endpoint_id

    with st.form("api_credential_form"):
        api_key = st.text_input(
            "LandingLens API Key",
            key="lnd_api_key",
            value=st.session_state["api_key"],
            help="If left empty, the free trial API key is used. The default key is a free trial key with a rate limit, i.e. X times per day.",
            type="password",
        )
        endpoint_id = None
        if render_endpoint_id:
            endpoint_id = st.text_input(
                "Cloud Deployment endpoint ID",
                key="lnd_endpoint_id",
                value=st.session_state.get("endpoint_id", ""),
            )

        submitted = st.form_submit_button("Save")
        if submitted:
            _update_credential(api_key, endpoint_id)
            st.info("API key is saved successfully")

    key = get_api_key_or_use_default()
    if key == get_default_api_key():
        st.info("The default API key (free trial) is used")
