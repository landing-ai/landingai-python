import logging
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

from landingai.common import Prediction
from landingai.postprocess import class_counts, class_map
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]


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
        predictions = self.image_predictions
        images = [pred.image for pred in predictions]
        image_paths = [str(pred.local_image_path) for pred in predictions]
        class_counts_all = {
            class_name: [0] * len(images) for class_name in self.class_map.values()
        }
        for i, res in enumerate(predictions):
            for count, class_name in res.class_counts.values():
                class_counts_all[class_name][i] = count

        data = {
            "image_path": image_paths,
        }
        data.update(class_counts_all)
        return pd.DataFrame(data)


def bulk_inference(image_paths: list[str], image_folder_path: str) -> ImageFolderResult:
    endpoint_id = st.session_state["endpoint_id"]
    api_key = st.session_state["api_key"]
    api_secret = st.session_state["api_secret"]
    predictor = Predictor(endpoint_id, api_key=api_key, api_secret=api_secret)
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
        and st.session_state.get("api_secret")
    )


def reset_states():
    st.session_state["image_paths"] = []
    st.session_state["result"] = []

st.header("Inference on Image Folder")
st.divider()
st.subheader("Select image folder")
local_image_folder = st.text_input(
    "Image Folder Path",
    key="image_folder_path",
    value="examples/output/streamlit",
    on_change=reset_states,
)

if local_image_folder:
    if not is_landing_credentials_set():
        st.error(
            "Please go to the config page and enter your CloudInference endpoint ID first."
        )
        st.stop()

    image_paths = []
    for ext in _SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(list(Path(local_image_folder).glob(f"*.{ext}")))
    if not image_paths:
        st.warning(
            f"No images found in {local_image_folder}. Supported image formats: {_SUPPORTED_IMAGE_FORMATS}"
        )
        st.stop()

    logging.info(
        f"Found {len(image_paths)} images in {local_image_folder}: {image_paths}"
    )
    with st.expander("Preview input images"):
        img_path = image_select(
            label=f"Total {len(image_paths)} images",
            images=image_paths,
            key="preview_input_images",
            captions=[f"{p.name}" for p in image_paths],
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
            "image_paths": image_paths,
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
                label=f"Total {len(image_paths)} images",
                images=img_with_preds,
                key="preview_pred_images",
                captions=[f"{p.name}" for p in image_paths],
                use_container_width=False,
            )
            img_path_str = str(img_path)
            st.image(img_path_str, caption=img_path_str)

        all_preds = []
        for pred in result.image_predictions:
            all_preds.extend(pred.predictions)
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
