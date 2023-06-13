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
    type="password",
    value=st.session_state.get("api_secret", ""),
)
endpoint_id = st.text_input(
    "CloudInference endpoint ID",
    key="lnd_endpoint_id",
    value=st.session_state.get("endpoint_id", ""),
)

snow_user = st.text_input(
    "Snowflake Username",
    key="snowflake_user",
    value=st.session_state.get("snow_user", ""),
)
snow_password = st.text_input(
    "Snowflake User Password",
    key="snowflake_password",
    type="password",
    value=st.session_state.get("snow_password", ""),
)
snow_account = st.text_input(
    "Snowflake Account",
    key="snowflake_account",
    value=st.session_state.get("snow_account", ""),
)
snow_warehouse = st.text_input(
    "Snowflake Warehouse",
    key="snowflake_warehouse",
    value=st.session_state.get("snow_warehouse", ""),
)
snow_database = st.text_input(
    "Snowflake Database",
    key="snowflake_database",
    value=st.session_state.get("snow_database", ""),
)
snow_schema = st.text_input(
    "Snowflake Schema",
    key="snowflake_schema",
    value=st.session_state.get("snow_schema", ""),
)

snow_config = (
    snow_user,
    snow_password,
    snow_account,
    snow_warehouse,
    snow_database,
    snow_schema,
)

# Sample project
# Key: v7b0hdyfj6271xy2o9lmiwkkcbdpvt1
# Secret: ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq3u1cgq
# 28d6279d-0db1-4beb-9803-fcae7a7c5df5

if "api_key" not in st.session_state:
    st.session_state["api_key"] = "v7b0hdyfj6271xy2o9lmiwkkcbdpvt1"
if "api_secret" not in st.session_state:
    st.session_state[
        "api_secret"
    ] = "ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq3u1cgq"
if "endpoint_id" not in st.session_state:
    st.session_state["endpoint_id"] = "28d6279d-0db1-4beb-9803-fcae7a7c5df5"
if "snow_user" not in st.session_state:
    st.session_state["snow_user"] = ""
if "snow_password" not in st.session_state:
    st.session_state["snow_password"] = ""
if "snow_account" not in st.session_state:
    st.session_state["snow_account"] = ""
if "snow_warehouse" not in st.session_state:
    st.session_state["snow_warehouse"] = ""
if "snow_database" not in st.session_state:
    st.session_state["snow_database"] = ""
if "snow_schema" not in st.session_state:
    st.session_state["snow_schema"] = ""
if "snow_config" not in st.session_state:
    st.session_state["snow_config"] = None


def save_config(snow_config, api_key, api_secret, endpoint_id):
    st.session_state["api_key"] = api_key
    st.session_state["api_secret"] = api_secret
    st.session_state["endpoint_id"] = endpoint_id
    (
        snow_user,
        snow_password,
        snow_account,
        snow_warehouse,
        snow_database,
        snow_schema,
    ) = snow_config
    snow_credential = SnowflakeCredential(
        user=snow_user,
        password=snow_password,
        account=snow_account,
    )
    snow_db_config = SnowflakeDBConfig(
        warehouse=snow_warehouse,
        database=snow_database,
        snowflake_schema=snow_schema,
    )
    st.session_state["snow_config"] = (snow_credential, snow_db_config)


# st.write("api_key:", st.session_state.api_key)
# st.write("api_secret:", st.session_state.api_secret)
# st.write("endpoint_id:", st.session_state.endpoint_id)

if st.button(
    "Save",
    on_click=save_config,
    args=(snow_config, api_key, api_secret, endpoint_id),
):
    st.info("Configuration saved successfully...")
st.divider()
