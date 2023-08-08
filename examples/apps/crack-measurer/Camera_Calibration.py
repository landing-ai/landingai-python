import streamlit as st
import math
import landingai.st_utils as lst

from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


if "points" not in st.session_state:
    st.session_state["points"] = []


def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def main():
    lst.setup_page("Measure Crack Dimensions")

    st.sidebar.title("Camera Calibration 📷")

    title_str = f"""
    <style>
    p.title {{
        font: bold {67}px "Times New Roman";
        color: red;
        text-align: center;
    }}
    </style>
    <p class="title">Camera Calibration 📷</p>
    """
    st.markdown(title_str, unsafe_allow_html=True)

    col1, col2 = st.columns((2, 5), gap="small")

    with col2:
        file_uploader_label = "If not known, take a picture of an object with \
                               known size using the same zoom as used for the \
                               crack photos, and click on the two ends of the \
                               object below"
        uploaded_file = st.file_uploader(
            file_uploader_label, type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img.thumbnail((512, 512))
            draw = ImageDraw.Draw(img)

            for point in st.session_state["points"]:
                coords = get_ellipse_coords(point)
                draw.ellipse(coords, fill="red")
            if len(st.session_state["points"]) == 2:
                draw.line(
                    st.session_state["points"],
                    fill="red",
                    width=5,
                )
            value = streamlit_image_coordinates(img, key="pil")

            if value is not None:
                point = value["x"], value["y"]
                if point not in st.session_state["points"]:
                    st.session_state["points"].append(point)
                    st.session_state["points"] = st.session_state["points"][-2:]
                    st.experimental_rerun()
    with col1:
        st.write("#")
        measure = st.number_input(
            "How man inches long is this object?", min_value=0.1, value=0.1
        )
        if len(st.session_state["points"]) == 2:
            dist = math.dist(
                st.session_state["points"][0], st.session_state["points"][1]
            )
            st.session_state["inch_to_pixels"] = dist / measure
            st.write("Pixels per inch: ", st.session_state["inch_to_pixels"])


if __name__ == "__main__":
    main()
