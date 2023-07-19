import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import math
import landingai.st_utils as lst


def _resize(
    img: Image,
    size: int) -> Image:
    """Resize image to given size and keep relative dimensions.

    Parameters
    ----------
    img : Image
        Input. The image to be resized.

    size : int
        The new size of the given image.

    """
    image = Image.open(img)
    try:
        image.thumbnail((size, size))
    except (OSError, SyntaxError):
        # Occurs when the image is not fully loaded. In this specific case, it 
        # occurs often when the user does something that reloads the page in
        # quick succession, such as tapping quickly on the image when selecting
        # coordinates or tapping quickly on the number input buttons (+/-)
        return "Image did not load properly"
    return image


def main():
    # Page Setup
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

    col1, col2 = st.columns((2, 5), gap = "small")
        
    number_input_tooltip = "Type a number and use the +/- buttons to \
                            increase or decrease the amount by 0.01"

    # Set up textbox in first column
    with col1: 
        number_input_label = "If known, input number of pixels per inch"

        if "inch_to_pixels" in st.session_state and \
           "user_inputted" in st.session_state: 
            st.session_state.value = st.session_state.inch_to_pixels
            del st.session_state.user_inputted
        else:
            st.session_state.value = 0.00
        for k in st.session_state.keys():
            st.session_state[k] = st.session_state[k]
        user_input_pixels_to_inch = st.number_input(
            number_input_label, 
            min_value=0.00, 
            value=float(st.session_state.value),
            key="b",
            help=number_input_tooltip)
        

        st.session_state.value = user_input_pixels_to_inch

        st.session_state.inch_to_pixels = user_input_pixels_to_inch

    # Set up file uploader in second column
    with col2:
        file_uploader_label = "If not known, take a picture of an object with \
                               known size using the same zoom as used for the \
                               crack photos, and click on the two ends of the \
                               object below"
        uploaded_file = st.file_uploader(file_uploader_label, 
                                         type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file

        image = ""
        if "uploaded_file" in st.session_state and \
            st.session_state.uploaded_file is not None:
            image = _resize(st.session_state.uploaded_file, 300)
            if isinstance(image, str): # If there is an issue loading the image
                st.write(image)
                st.experimental_rerun()


        if not isinstance(image, str):
            _, col4 = st.columns((1, 5))

            with col4:
                # Allow user to click image and get coordinates of user clicks
                coords = streamlit_image_coordinates(image)
                
            if user_input_pixels_to_inch:
                st.write("Delete the value entered above if you would like to \
                         select points in the uploaded picture")
            else:
                with col1:
                    st.write("#")
                    measure_num_input = "How many inches long is this object?"
                    measure = st.number_input(measure_num_input, 
                                              min_value=0.01, 
                                              help=number_input_tooltip)
                    
                    if coords and "coords" in st.session_state and \
                        st.session_state.coords and \
                        int(user_input_pixels_to_inch) == 0:                   
                        # Calculate distance between two clicked points
                        p1 = (coords['x'], coords['y'])
                        p2 = (st.session_state.coords['x'],
                              st.session_state.coords['y'])
                        dist = math.dist(p1, p2)
                        st.session_state.inch_to_pixels = dist / measure

                    st.write("Coordinate position clicked in image:")
                    st.write(coords)
                    st.session_state.coords = coords

    # Set up inch_to_pixels display                                              
    if "inch_to_pixels" in st.session_state:
        if st.session_state.inch_to_pixels <= 0 and not isinstance(image, str):
            st.write("Inch to pixel ratio must be greater than 0.")
        else:
            if isinstance(st.session_state.inch_to_pixels, float):
                rounded_pixel_ratio = round(st.session_state.inch_to_pixels, 2)
                st.session_state.inch_to_pixels = rounded_pixel_ratio

            inch_to_pixel_str = f"""
            <style>
            p.inch_to_pixels {{
                font: bold {30}px Courier;
                color: red;
                text-align: center;
            }}
            </style>
            <p class="inch_to_pixels">Inch to pixel ratio: 
                        {st.session_state.inch_to_pixels}</p>
            """
            st.write("#")
            st.markdown(inch_to_pixel_str, unsafe_allow_html=True)
            if user_input_pixels_to_inch > 0.0:
                st.session_state.user_inputted = True


if __name__ == "__main__":
    main()
