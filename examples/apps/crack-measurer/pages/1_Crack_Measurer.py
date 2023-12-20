import math

import cv2
import numpy as np
import streamlit as st
from centerline.geometry import Centerline
from PIL import Image
from shapely.geometry import Polygon
from skimage.morphology import medial_axis

import landingai.st_utils as lst
from landingai.common import decode_bitmap_rle
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions


def find_endpoints(start0, start1, max_rows, max_cols, binary_image):
    start = None
    end = None
    for row in range(int(start0), -1, -1):
        col = int(start1) + 3
        if row >= max_rows - 2:
            break
        if row >= 0 and row < max_rows and binary_image[col, row] != 0:
            start = (row, col)
            break
    for row in range(int(start0), max_cols - 1):
        col = int(start1) + 3
        if row >= 0 and row < max_rows and binary_image[col, row] != 0:
            end = (row, col)
            break
    return start, end


def extend_line_to_binary(start_point, end_point, binary_image, org_image):
    # Calculate slope and intercept of the line
    if end_point[0] - start_point[0] != 0:
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    else:
        slope = float("inf")

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
        if row >= 0 and row < rows and binary_image[col, row] != 0:
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
        a, b = find_endpoints(start_point[0], start_point[1], rows, cols, binary_image)
        extended_start_point = a if a is not None else extended_start_point
        extended_end_point = b if b is not None else extended_end_point

    if start_point[0] == 0:
        a, b = find_endpoints(start_point[1], start_point[0], cols, rows, binary_image)
        extended_start_point = a if a is not None else extended_start_point
        extended_end_point = b if b is not None else extended_end_point

    distance = math.dist(extended_start_point, extended_end_point)

    # Draw the extended line on the binary image
    rgb_image = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
    output_image = cv2.line(
        rgb_image, extended_start_point, extended_end_point, (255, 0, 0), thickness=5
    )

    return (output_image, distance)


def width(contours, seg_mask_channel, final_img, new_arr):
    delta = 3

    new_im = seg_mask_channel

    test_arr = np.array(np.zeros_like(seg_mask_channel), dtype=np.uint8)

    cv2.drawContours(
        test_arr, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1
    )

    medial, distance = medial_axis(test_arr, return_distance=True)

    skeleton_int_array1 = np.asarray(medial, dtype=np.uint8)
    skeleton_int_array1 = np.expand_dims(skeleton_int_array1, axis=-1)
    skeleton_int_array1 *= 255
    skeleton_int_array1 = skeleton_int_array1.squeeze()

    med_contours, _ = cv2.findContours(
        skeleton_int_array1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(
        new_im, [max(med_contours, key=cv2.contourArea)], -1, 255, thickness=-1
    )
    med_pts = [v[0] for v in max(med_contours, key=cv2.contourArea)]

    # get point with maximal distance from medial axis
    max_idx = np.argmax(distance)
    max_pos = np.unravel_index(max_idx, distance.shape)
    coords = np.array([max_pos[1] - 1, max_pos[0]])

    # interpolate orthogonal of medial axis at coords
    idx = next(
        (i for i, v in enumerate(med_pts) if (v == coords).all()), 100
    )  # FIXXXXX
    px1, py1 = med_pts[(idx - delta) % len(med_pts)]
    px2, py2 = med_pts[(idx + delta) % len(med_pts)]
    vector = np.array([px2 - px1, py2 - py1])
    orth = np.array([-vector[1], vector[0]])

    # intersect orthogonal with crack and get contour
    orth_img = np.zeros(final_img.shape, dtype=np.uint8)
    cv2.line(
        orth_img, tuple(coords + orth), tuple(coords - orth), color=255, thickness=1
    )
    gap_img = cv2.bitwise_and(orth_img, new_im)
    gap_contours, _ = cv2.findContours(
        np.asarray(gap_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gap_pts = [v[0] for v in gap_contours[0]]

    # determine the end points of the gap contour by negative dot product
    n = len(gap_pts)
    gap_ends = [
        p
        for i, p in enumerate(gap_pts)
        if np.dot(p - gap_pts[(i - 1) % n], gap_pts[(i + 1) % n] - p) < 0
    ]
    if gap_ends[0][0] > gap_ends[1][0]:
        gap_ends[0], gap_ends[1] = gap_ends[1], gap_ends[0]
    new1_im = new_im.copy().squeeze()
    return extend_line_to_binary(gap_ends[0], gap_ends[1], new_arr, new1_im)


predictor = Predictor(
    "119abed3-517d-465c-b85a-e0e721210e19",
    api_key="land_sk_3D0h1C58KRq6yN5KHaWjqOyp81khgFQxydEmGdHPJy6j9GbvY8",
)


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

    uploaded_file = st.file_uploader("Upload crack image")
    if uploaded_file is not None and "inch_to_pixels" in st.session_state:
        image = Image.open(uploaded_file).convert("RGB")
        seg_pred = predictor.predict(image)
        color_dict = {"crack": "red"}
        image_with_preds = overlay_predictions(
            seg_pred, np.asarray(image), {"color_map": color_dict}
        )
        st.image(image_with_preds, caption="Segmentation Predictions")
        # PIL is WxH, numpy is HxW
        image_shape = image.size[::-1]

        if len(seg_pred) == 0:
            st.error("No cracks detected in the image")
            return

        flattened_bitmap = decode_bitmap_rle(
            seg_pred[0].encoded_mask, seg_pred[0].encoding_map
        )
        seg_mask_channel = np.array(flattened_bitmap, dtype=np.uint8).reshape(
            image_shape
        )
        seg_mask_channel *= 255

        # Find the contours in the line_array
        contours, _ = cv2.findContours(
            seg_mask_channel.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        max_contour = max(contours, key=cv2.contourArea)

        # Perform contour smoothing using the Douglas-Peucker algorithm
        epsilon = 0.001 * cv2.arcLength(max_contour, True)
        smoothed_contour = cv2.approxPolyDP(max_contour, epsilon, True)

        smoothed_array = smoothed_contour.squeeze()

        poly = Polygon(smoothed_array)
        cl = Centerline(poly)

        new_arr = np.zeros_like(image_with_preds)

        cv2.drawContours(new_arr, [smoothed_contour], -1, (0, 0, 255), 2)
        new_arr = cv2.cvtColor(new_arr, cv2.COLOR_BGR2GRAY)

        c_array = np.zeros((len(cl.geometry.geoms), 4), dtype=float)
        for i, geom in enumerate(cl.geometry.geoms):
            cords = list(geom.coords)
            c_array[i] = [cords[0][0], cords[0][1], cords[-1][0], cords[-1][1]]

        final_img = np.zeros(image_shape, dtype=np.uint8)
        for seg in c_array:
            start_x = int(seg[0])
            start_y = int(seg[1])
            end_x = int(seg[2])
            end_y = int(seg[3])
            final_img[start_y : end_y + 1, start_x : end_x + 1] = 255
        st.image(
            final_img,
            caption="Line used to calculate length of longest crack segment",
        )

        # Assume there's only one contour
        length_pixels = 0
        for contour in contours:
            length_pixels += cv2.arcLength(contour, True)

        length_pixels = length_pixels
        width_image, width_pixels = width(
            contours, seg_mask_channel, final_img, new_arr
        )
        st.image(
            width_image,
            "Largest perpendicular width of crack found in longest crack segment",
        )

        st.write(
            "Predicted crack length in inches: "
            + str(length_pixels / 2 / st.session_state.inch_to_pixels)
        )
        st.write(
            "Predicted crack width in inches: "
            + str(width_pixels / 2 / st.session_state.inch_to_pixels)
        )

    if "inch_to_pixels" not in st.session_state:
        st.write("Please calibrate your camera first")


if __name__ == "__main__":
    main()
