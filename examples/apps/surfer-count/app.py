import streamlit as st

st.title("Topanga Beach Surfer Counter")
st.write(
    "This application will grab the latest 10s clip of surfers from the Topanga Beach surf cam"
    "and count the number of surfers there."
)
st.write("Please enter your LandingLens API key and Cloud Inference Endpoint ID.")
api_key = st.text_input(
    "LandingLens API Key", value=st.session_state.get("api_key", "")
)
endpoint_id = st.text_input(
    "Cloud Inference Endpoint ID",
    value=st.session_state.get("endpoint_id", ""),
)


def save(api_key: str, endpoint_id: str):
    st.session_state["api_key"] = api_key
    st.session_state["endpoint_id"] = endpoint_id


st.button("Save", on_click=save(api_key, endpoint_id))
