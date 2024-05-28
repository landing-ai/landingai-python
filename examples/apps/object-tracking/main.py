import streamlit as st

st.title("LandingAI Traffic Tracking")
st.write(
    "This application will grab the latest 10 second traffic clip of the Pacific Coast Highway"
    "going to Malibu and count the number of cars going northbound, southbound and parked."
)
st.write("Please enter your LandingLens API credentials and CloudInference Endpoint ID")
api_key = st.text_input(
    "LandingLens API Key", value=st.session_state.get("api_key", "")
)
endpoint_id = st.text_input(
    "CloudInference Endpoint ID",
    value=st.session_state.get("endpoint_id", ""),
)


def save(api_key, endpoint_id):
    st.session_state["api_key"] = api_key
    st.session_state["endpoint_id"] = endpoint_id


st.button("Save", on_click=save(api_key, endpoint_id))
