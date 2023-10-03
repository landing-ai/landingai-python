import streamlit as st


st.title("")
# api_key = st.text_input(
#     "LandingLens API Key", value=st.session_state.get("api_key", "")
# )
# endpoint_id = st.text_input(
#     "Cloud Endpoint ID", value=st.session_state.get("endpoint_id", "")
# )


api_key = "land_sk_JkygHlib8SgryZUgumM6r8GWYfQqiKdE36xDzo4K85fDihpnuG"
endpoint_id = "7e8c1f16-947f-45cd-9f5d-c5bdf8791126"


def save(api_key: str, endpoint_id: str):
    st.session_state["api_key"] = api_key
    st.session_state["endpoint_id"] = endpoint_id


st.button("Save", on_click=save(api_key, endpoint_id))
