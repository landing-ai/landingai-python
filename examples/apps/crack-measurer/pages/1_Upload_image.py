import streamlit as st
import landingai.st_utils as lst


def main():
    # Page Setup
    lst.setup_page("Image Upload")

    title_str = f"""
    <style>
    p.title {{
        font: bold {75}px "Times New Roman";
        color: green;
        text-align: center;
    }}
    </style>
    <p class="title">Upload Image 🌆</p>
    """
    st.markdown(title_str, unsafe_allow_html=True)

    st.sidebar.title("Upload Image 🌆")

    
    # Setup file uploader
    crack_image = st.file_uploader("Upload crack image")

    # Make sure the uploaded image continues in the current session even when
    # switching between pages, until a new image is uploaded.
    if crack_image and \
       (("image" in st.session_state and crack_image != st.session_state.image) \
       or ("image" not in st.session_state)):
        st.session_state.image = crack_image
        st.image(crack_image)
    elif "image" in st.session_state:
        st.image(st.session_state.image)

if __name__ == "__main__":
    main() 
