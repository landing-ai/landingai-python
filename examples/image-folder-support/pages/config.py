import logging

import streamlit as st

from landingai.storage.snowflake import SnowflakeCredential, SnowflakeDBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

st.subheader("Configuration")
st.write(
    "Please enter your Snowflake credential and LandingLens API key and secret below."
)


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

# 2eifa2kg55nefqi2xpr6lzhlk0orcyy
# wf9wyohentdfwwxsrpv4dhulz7qopirmgf7pbh6vg2yqs4whh7vp9dd3p0vteb
# dfa79692-75eb-4a48-b02e-b273751adbae

# Sample project
# Key: v7b0hdyfj6271xy2o9lmiwkkcbdpvt1
# Secret: ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq3u1cgq
# 036d86dc-f08d-4eb8-ac07-70e6bbf2ff56

# Automatically fill a default value for convenience
if "api_key" not in st.session_state:
    st.session_state["api_key"] = "v7b0hdyfj6271xy2o9lmiwkkcbdpvt1"
if "api_secret" not in st.session_state:
    st.session_state[
        "api_secret"
    ] = "ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq3u1cgq"
if "endpoint_id" not in st.session_state:
    st.session_state["endpoint_id"] = "036d86dc-f08d-4eb8-ac07-70e6bbf2ff56"


def save_config(api_key, api_secret, endpoint_id):
    st.session_state["api_key"] = api_key
    st.session_state["api_secret"] = api_secret
    st.session_state["endpoint_id"] = endpoint_id


if st.button(
    "Save",
    on_click=save_config,
    args=(api_key, api_secret, endpoint_id),
):
    st.info("Configuration saved successfully...")
st.divider()
