import logging
import math
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import plotly.express as px
import streamlit as st
from PIL.Image import Image
from streamlit_image_select import image_select

from landingai.common import Prediction, SegmentationPrediction
from landingai.pipeline.image_source import ImageFolder
from landingai.postprocess import (
    class_counts,
    class_map,
    segmentation_class_pixel_coverage,
)
from landingai.predict import Predictor
from landingai.st_utils import (
    check_api_credentials_set,
    check_endpoint_id_set,
    render_api_config_form,
    setup_page,
)
from landingai.visualize import overlay_predictions

setup_page(page_title="LandingLens Image Folder App")

_SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "tiff"]


@dataclass(frozen=True)
class ImageInferenceResult:
    local_image_path: Path
    image: np.ndarray
    predictions: list[Prediction]
    img_with_preds_path: Path

    @cached_property
    def class_counts(self) -> dict[int, tuple[int, str]]:
        return class_counts(self.predictions)

    @cached_property
    def image_with_predictions(self) -> Image:
        return overlay_predictions(self.predictions, self.image)


@dataclass(frozen=True)
class ImageFolderResult:
    image_folder: Path
    image_predictions: list[ImageInferenceResult]

    @cached_property
    def class_map(self) -> dict[int, str]:
        all_predictions = []
        for res in self.image_predictions:
            all_predictions.extend(res.predictions)
        return class_map(all_predictions)

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        if not self.image_predictions:
            return pd.DataFrame()
        class_names = self.class_map.values()
        data = []
        for pred_result in self.image_predictions:
            pred_json = {
                "image_path": str(pred_result.local_image_path),
                "total_pixel_count": math.prod(pred_result.image.shape[:2]),
            }
            for cls_name in class_names:
                if type(pred_result.predictions[0]) == SegmentationPrediction:
                    coverage_by_class = segmentation_class_pixel_coverage(
                        pred_result.predictions, "relative"
                    )
                    pixel_count_by_class = segmentation_class_pixel_coverage(
                        pred_result.predictions, "absolute"
                    )
                else:
                    coverage_by_class = {}
                    pixel_count_by_class = {}

                cls_cnt_map = {
                    cls_name: val
                    for (val, cls_name) in pred_result.class_counts.values()
                }
                pixel_cnt_map = {
                    cls_name: val for (val, cls_name) in pixel_count_by_class.values()
                }
                coverage_map = {
                    cls_name: val for (val, cls_name) in coverage_by_class.values()
                }
                pred_json[cls_name] = {
                    "class_count": cls_cnt_map.get(cls_name, 0),
                    "pixel_count": pixel_cnt_map.get(cls_name, 0),
                    "pixel_coverage": coverage_map.get(cls_name, 0.0),
                }
            data.append(pred_json)

        df = pd.json_normalize(data, max_level=1)
        return df


def bulk_inference(image_paths: list[str], image_folder_path: str) -> ImageFolderResult:
    endpoint_id = st.session_state["endpoint_id"]
    api_key = st.session_state["api_key"]
    predictor = Predictor(endpoint_id, api_key=api_key)
    images = [np.asarray(PIL.Image.open(p)) for p in image_paths]
    local_cache_dir = Path(tempfile.mkdtemp())
    pbar = st.progress(0, text="Running inferences...")
    result = []
    percent_complete = 1 / len(images)
    for i, (img, img_path) in enumerate(zip(images, image_paths)):
        preds = predictor.predict(img)
        img_with_preds = overlay_predictions(preds, img)
        filename = Path(img_path).stem
        img_with_preds_path = local_cache_dir / f"{filename}.jpg"
        img_with_preds.save(str(img_with_preds_path))
        result.append(
            ImageInferenceResult(
                local_image_path=Path(img_path),
                image=img,
                predictions=preds,
                img_with_preds_path=img_with_preds_path,
            )
        )
        percent_progress = percent_complete * (i + 1)
        text = f"Running inferences... {percent_progress:.1%} completed"
        pbar.progress(percent_progress, text=text)
    result = ImageFolderResult(Path(image_folder_path), result)
    st.session_state["result"] = result
    return result


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def is_landing_credentials_set():
    return (
        st.session_state.get("endpoint_id")
        and st.session_state.get("api_key")
    )


def reset_states():
    st.session_state["image_paths"] = []
    st.session_state["result"] = []


st.sidebar.title("API Configuration")
with st.sidebar:
    render_api_config_form(render_endpoint_id=True)

st.header("Inference on Image Folder")
st.divider()
st.subheader("Select image folder")
local_image_folder = st.text_input(
    "Image Folder Path",
    key="image_folder_path",
    value="",
    on_change=reset_states,
)

if local_image_folder:
    check_api_credentials_set()
    check_endpoint_id_set()

    folder_source = ImageFolder(source=local_image_folder)
    if len(folder_source) == 0:
        st.warning(
            f"No images found in {local_image_folder}. Supported image formats: {_SUPPORTED_IMAGE_FORMATS}"
        )
        st.stop()

    logging.info(
        f"Found {len(folder_source)} images in {local_image_folder}: {folder_source}"
    )
    with st.expander("Preview input images"):
        img_path = image_select(
            label=f"Total {len(folder_source)} images",
            images=folder_source.image_paths,
            key="preview_input_images",
            captions=[f"{Path(p).name}" for p in folder_source.image_paths],
            use_container_width=False,
        )
        img_path_str = str(img_path)
        st.image(img_path_str, caption=img_path_str)

    st.divider()
    st.subheader("Run inferences")
    request_to_run_inference = st.button(
        "Run inferences",
        on_click=bulk_inference,
        kwargs={
            "image_paths": folder_source.image_paths,
            "image_folder_path": local_image_folder,
        },
        type="secondary",
    )
    result = st.session_state.get("result", None)
    if result:
        img_with_preds = [
            str(pred.img_with_preds_path) for pred in result.image_predictions
        ]
        with st.expander("Preview prediction results"):
            img_path = image_select(
                label=f"Total {len(folder_source)} images",
                images=img_with_preds,
                key="preview_pred_images",
                captions=[
                    f"{p.img_with_preds_path.name}" for p in result.image_predictions
                ],
                use_container_width=False,
            )
            img_path_str = str(img_path)
            st.image(img_path_str, caption=img_path_str)

            for res in result.image_predictions:
                if str(res.img_with_preds_path) == img_path_str:
                    preds = res.predictions
                    if not preds or not isinstance(preds[0], SegmentationPrediction):
                        continue
                    coverage = segmentation_class_pixel_coverage(preds).values()
                    labels = [val[1] for val in coverage]
                    sizes = [val[0] for val in coverage]
                    unclassified_pixels = 1 - sum(sizes)
                    sizes.append(unclassified_pixels)
                    labels.append("unclassified")
                    fig = px.pie(
                        values=sizes,
                        names=labels,
                        title="Predicted pixel distribution over a single image",
                    )
                    st.plotly_chart(fig)
                    colored_masks = [pred.decoded_colored_mask for pred in preds]
                    label_names = [pred.label_name for pred in res.predictions]
                    selected_label = st.radio(
                        "Select a label to visualize", label_names, index=0
                    )
                    selected_mask = colored_masks[label_names.index(selected_label)]
                    st.image(selected_mask)

        all_preds = []
        for pred in result.image_predictions:
            all_preds.extend(pred.predictions)

        if not all_preds:
            st.warning("No predictions found.")
            st.stop()

        if isinstance(all_preds[0], SegmentationPrediction):
            coverage = segmentation_class_pixel_coverage(all_preds).values()
            labels = [val[1] for val in coverage]
            sizes = [val[0] for val in coverage]
            unclassified_pixels = 1 - sum(sizes)
            sizes.append(unclassified_pixels)
            labels.append("unclassified")
            fig = px.pie(
                values=sizes,
                names=labels,
                title="Predicted pixel distribution over all images",
            )
            st.plotly_chart(fig)
        else:
            class_cnts = class_counts(all_preds).values()
            cnts = [cnt_name[0] for cnt_name in class_cnts]
            names = [cnt_name[1] for cnt_name in class_cnts]
            fig = px.pie(
                values=cnts,
                names=names,
                title="Predicted class distribution over all images",
            )
            st.plotly_chart(fig)

        csv = convert_df(result.dataframe)
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            key="download-csv",
        )

st.divider()
