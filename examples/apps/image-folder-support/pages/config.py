import streamlit as st


st.subheader("Configuration")
st.write("Please enter your LandingLens API key and secret below.")


api_key = st.text_input(
    "LandingLens API Key", key="lnd_api_key", value=st.session_state.get("api_key", "")
)
api_secret = st.text_input(
    "LandingLens API Secret",
    key="lnd_api_secret",
    value=st.session_state.get("api_secret", ""),
)
endpoint_id = st.text_input(
    "CloudInference endpoint ID",
    key="lnd_endpoint_id",
    value=st.session_state.get("endpoint_id", ""),
)

# Automatically fill a default value for convenience
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "api_secret" not in st.session_state:
    st.session_state["api_secret"] = ""
if "endpoint_id" not in st.session_state:
    st.session_state["endpoint_id"] = ""


def save_config(api_key, api_secret, endpoint_id):
    st.session_state["api_key"] = api_key
    st.session_state["api_secret"] = api_secret
    st.session_state["endpoint_id"] = endpoint_id

if st.button("Save", on_click=save_config,
    args=(api_key, api_secret, endpoint_id),
):
    st.info("Configuration saved successfully...")
st.divider()
