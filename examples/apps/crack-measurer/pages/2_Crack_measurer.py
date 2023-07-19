from PIL import Image
import streamlit as st
import cv2
import math
import numpy as np
from landingai.visualize import overlay_predictions
from landingai.common import decode_bitmap_rle
import landingai.st_utils as lst
import base64
import io
from centerline.geometry import Centerline
from shapely.geometry import Polygon
from skimage.morphology import medial_axis
from landingai.predict import Predictor




def extend_line_to_binary(start_point, end_point, binary_image, org_image):
    # Calculate slope and intercept of the line
    if end_point[0] - start_point[0] != 0:
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    else:
        slope = float('inf')
 
    if slope == 0:
        slope = 0.001
    # Extend the line until it reaches the first point in the binary image
    rows, cols = binary_image.shape[:2]



    # Extend line in the negative direction
    extended_start_point = start_point
    for col in range(int(start_point[1]), -1, -1):
        row = int(start_point[0] + (col - start_point[1]) / slope)
        if row >= rows - 2:
            break
        if row >= 0 and row < rows and (binary_image[col, row] != 0):
            extended_start_point = (row, col)
            break
        
    # Extend line in the positive direction
    extended_end_point = end_point
    for col in range(int(start_point[1]), cols - 1):
        row = int(start_point[0] + (col - start_point[1]) / slope)
        if row >= 0 and row < rows and binary_image[col, row] != 0:
            extended_end_point = (row, col)
            break




    if start_point[1] == 0:
        for row in range(int(start_point[0]), -1, -1):
            col = int(start_point[1]) + 3
            if row >= rows - 2:
                break
            if row >= 0 and row < rows and (binary_image[col, row] != 0):
                extended_start_point = (row, col)
                break
        for row in range(int(start_point[0]), cols - 1):
            col = int(start_point[1]) + 3
            if row >= 0 and row < rows and binary_image[col, row] != 0:
                extended_end_point = (row, col)
                break

    if start_point[0] == 0:
        for col in range(int(start_point[1]), -1, -1):
            row = int(start_point[0]) + 3
            if row >= rows - 2:
                break
            if row >= 0 and row < rows and (binary_image[col, row] != 0):
                extended_start_point = (row, col)
                break
        for col in range(int(start_point[1]), cols - 1):
            row = int(start_point[0]) + 3
            if row >= 0 and row < rows and binary_image[col, row] != 0:
                extended_end_point = (row, col)
                break

    distance = math.dist(extended_start_point, extended_end_point)

    # Draw the extended line on the binary image
    rgb_image = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
    #extended_end_point = (1920, 1894)
    output_image = cv2.line(rgb_image, extended_start_point, extended_end_point, (255, 0, 0), thickness=5)

    

    return (output_image, distance)



def width(contours, seg_mask_channel, final_img, new_arr):
    delta = 3

    new_im = seg_mask_channel
    
    test_arr = np.array(np.zeros_like(seg_mask_channel), dtype=np.uint8).reshape(
        Image.open(st.session_state.image).size
    )

    cv2.drawContours(test_arr, [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)


    medial, distance = medial_axis(test_arr, return_distance=True)




    skeleton_int_array1 = np.asarray(medial, dtype = np.uint8)
    skeleton_int_array1 = np.expand_dims(skeleton_int_array1, axis=-1)
    skeleton_int_array1 *= 255
    skeleton_int_array1 = skeleton_int_array1.squeeze()




    med_contours, _ = cv2.findContours(skeleton_int_array1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    cv2.drawContours(new_im, [max(med_contours, key = cv2.contourArea)], -1, 255, thickness=-1)
    med_pts = [v[0] for v in max(med_contours, key = cv2.contourArea)]

    

    
    # get point with maximal distance from medial axis
    max_idx = np.argmax(distance)
    max_pos = np.unravel_index(max_idx, distance.shape)
    coords = np.array([max_pos[1]-1, max_pos[0]])

    # interpolate orthogonal of medial axis at coords
    idx = next((i for i, v in enumerate(med_pts) if (v == coords).all()), 100) # FIXXXXX
    px1, py1 = med_pts[(idx-delta) % len(med_pts)]
    px2, py2 = med_pts[(idx+delta) % len(med_pts)]
    vector = np.array([px2 - px1, py2 - py1])
    orth = np.array([-vector[1], vector[0]])

    # intersect orthogonal with crack and get contour
    orth_img = np.zeros(final_img.shape, dtype=np.uint8)
    cv2.line(orth_img, tuple(coords + orth), tuple(coords - orth), color=255, thickness=1)
    gap_img = cv2.bitwise_and(orth_img, new_im)
    gap_contours, _ = cv2.findContours(np.asarray(gap_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gap_pts = [v[0] for v in gap_contours[0]]

    # determine the end points of the gap contour by negative dot product
    n = len(gap_pts)
    gap_ends = [
        p for i, p in enumerate(gap_pts)
        if np.dot(p - gap_pts[(i-1) % n], gap_pts[(i+1) % n] - p) < 0
    ]
    if (gap_ends[0][0] > gap_ends[1][0]):
        gap_ends[0], gap_ends[1] = gap_ends[1], gap_ends[0]
    new1_im = new_im.copy().squeeze()
    return extend_line_to_binary(gap_ends[0], gap_ends[1], new_arr, new1_im)
    #cv2.line(other_im, gap_ends[0], gap_ends[1], color=(0, 0, 255), thickness=1)


def main():
    lst.setup_page("Measure Crack Dimensions")

    html_str = f"""
    <style>
    p.a {{
    font: bold {50}px "Times New Roman";
    color: blue;
    text-align: center;
    }}
    </style>
    <p class="a">Measure Crack Dimensions 📏</p>
    """
    st.markdown(html_str, unsafe_allow_html=True)

    st.sidebar.title("Measure Crack Dimensions 📏")

    if "inch_to_pixels" in st.session_state:
        if "image" in st.session_state:
            if "img_size" not in st.session_state or "image_copy" in st.session_state and st.session_state.image_copy != st.session_state.image:
                image = Image.open(st.session_state.image).convert("RGB")
                
                st.session_state.img_size = image.size
                decoded_data = base64.b64decode(st.session_state.image.getvalue())
                buffer = io.BytesIO(decoded_data)
                image.save(buffer, format='JPEG')

                # Setup predictor with apikey, apisecret, and endpoint id.
                predictor = Predictor("5ab54748-8891-4ea1-8b9e-b4942de3b2dd", "i65k8vckxapnicpvvulxzv52i23rihs", "ksevngg38pz5mqqgpfhcvkkdagw67zf4gsj3wgfx89exhwksg2oh1zmfzqvveh")

                seg_pred = predictor.predict(image)
                st.session_state.seg_pred = seg_pred

                color_dict = { "Crack": "red" }
                img_with_preds = overlay_predictions(seg_pred, np.asarray(image), {"color_map": color_dict})
                st.session_state.img_with_preds = img_with_preds
                st.session_state.image_copy = st.session_state.image
        else:
            st.session_state.image_choice_msg = True
            st.write("Go choose an image")
    
        if "img_size" in st.session_state and "img_with_preds" in st.session_state:
            img_width, img_height = st.session_state.img_size
            st.write(f"Image size : {img_width} x {img_height}")
            st.image(st.session_state.img_with_preds, caption="Segmentation Inference")

            try:
                flattened_bitmap = decode_bitmap_rle(st.session_state.seg_pred[0].encoded_mask, st.session_state.seg_pred[0].encoding_map)

                seg_mask_channel = np.array(flattened_bitmap, dtype=np.uint8).reshape(
                    st.session_state.img_size
                )
                seg_mask_channel = seg_mask_channel * 255

                # Find the contours in the line_array
                contours, _ = cv2.findContours(seg_mask_channel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_contour = max(contours, key = cv2.contourArea)


                # Perform contour smoothing using the Douglas-Peucker algorithm
                epsilon = 0.001 * cv2.arcLength(max_contour, True)
                smoothed_contour = cv2.approxPolyDP(max_contour, epsilon, True)
                
                smoothed_array = smoothed_contour.squeeze()

                poly = Polygon(smoothed_array)
                cl = Centerline(poly)


                new_arr = np.zeros_like(st.session_state.img_with_preds)
                
            
                cv2.drawContours(new_arr, [smoothed_contour], -1, (0, 0, 255), 2)
                new_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY) 

                c_array = np.zeros((len(cl.geometry.geoms),4), dtype=float)
                for i, l in enumerate(cl.geometry.geoms):
                    ll = list(l.coords)
                    c_array[i] = [ll[0][0], ll[0][1], ll[-1][0], ll[-1][1]]
                final_img = np.zeros(st.session_state.img_size, dtype=np.uint8)
                for seg in c_array:
                    start_x = int(seg[0])
                    start_y = int(seg[1])
                    end_x = int(seg[2])
                    end_y = int(seg[3])
                    final_img[start_y:end_y+1, start_x:end_x+1] = 255
                st.image(final_img, caption="Line used to calculate length of longest crack segment")

                # Assume there's only one contour
                length_pixels = 0
                for contour in contours:
                    length_pixels += cv2.arcLength(contour, True)

                st.session_state.length_pixels = length_pixels
                width_image, st.session_state.width_pixels = width(contours, seg_mask_channel, final_img, new_arr)
                st.image(width_image, "Largest perpendicular width of crack found in longest crack segment")
            except IndexError:
                st.write("No crack found")
                if "length_pixels" in st.session_state:
                    del st.session_state.length_pixels
                if "width_pixels" in st.session_state:
                    del st.session_state.width_pixels
            
        else:
            if "image_choice_msg" not in st.session_state:
                st.write("An error occurred. Please refresh page and try again.")
     

        if "length_pixels" in st.session_state and "inch_to_pixels" in st.session_state:
            st.write("Predicted crack length in pixels: " + str(st.session_state.length_pixels / 2))
            st.write("Predicted crack length in inches: " + str(st.session_state.length_pixels / 2 / int(st.session_state.inch_to_pixels)))
            st.write("Predicted crack width in pixels: " + str(st.session_state.width_pixels / 2))
            st.write("Predicted crack width in inches: " + str(st.session_state.width_pixels / 2 / int(st.session_state.inch_to_pixels)))
    else:
        st.write("Please calibrate your camera first")

if __name__ == "__main__":
    main() 