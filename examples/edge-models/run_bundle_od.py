# python run_bundle_od.py published_model_bundle.zip image.jpg
# python run_bundle_od.py published_model_bundle.zip image.jpg /usr/lib/libvx_delegate.so
import json
import time
import zipfile
import argparse
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

IMAGENET_DEFAULT_MEAN = np.array([123.675, 116.28, 103.53])
IMAGENET_DEFAULT_STD = np.array([58.395, 57.12, 57.375])


def preprocess(img: np.ndarray, input_details: list[dict]) -> np.ndarray:
    """Preprocess the image for the model"""
    # Resize to the input shape
    print("Model input shape:", input_details[0]["shape"])
    H, W = input_details[0]["shape"][1:3]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    # Normalize
    img = (img - IMAGENET_DEFAULT_MEAN[None, None]) / IMAGENET_DEFAULT_STD[None, None]
    # Transform from HWC to BHWC format
    img = img[None]
    print("New image shape:", img.shape)
    # Convert to int8 if needed
    input_type = input_details[0]["dtype"]
    if input_type == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        img = normalize_int8(img, input_scale, input_zero_point)
    return img


def normalize_int8(
    np_image: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    np_image = np_image / scale + zero_point
    np_image = np.clip(np_image, -128, 127)
    return np_image.astype(np.int8)


def denormalize_int8_scores(
    scores: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    scores = (scores.astype(np.float32) - zero_point) * scale
    scores = np.clip(scores, 0, 1)
    return scores.astype(np.float32)


def denormalize_int8_labels(
    labels: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    return ((labels.astype(np.int32) - zero_point) * scale).astype(np.int32)


def read_img_rgb(img_path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def get_tflite_interpreter(
    model_path: str, ext_delegate: str | None
) -> tuple[tflite.Interpreter, list[dict], list[dict]]:
    if ext_delegate is not None:
        ext_delegate_options = None
        print(
            "Loading external delegate from {} with args: {}".format(
                ext_delegate, ext_delegate_options
            )
        )
        ext_delegate = [tflite.load_delegate(ext_delegate, ext_delegate_options)]

    interpreter = tflite.Interpreter(
        model_path=model_path, experimental_delegates=ext_delegate
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def predict(
    interpreter: tflite.Interpreter,
    input_details: list[dict],
    output_details: list[dict],
    img: np.ndarray,
) -> list[np.ndarray]:
    t0 = time.time()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    t1 = time.time()
    inference_time = t1 - t0
    print(f"Inference time: {inference_time}")
    predictions = []
    for output_layer in output_details:
        prediction = interpreter.get_tensor(output_layer["index"])
        predictions.append(prediction)
    return predictions


def run_inference(
    interpreter: tflite.Interpreter,
    input_details: list[dict],
    output_details: list[dict],
    img: np.ndarray,
    defect_map: dict[int, str],
) -> dict[str, Any]:
    """Run the model on the given preprocessed input"""
    predictions = predict(interpreter, input_details, output_details, img)

    output_dtypes, output_scales, output_zero_points = [], [], []
    for layer in output_details:
        output_dtypes.append(layer["dtype"])
        scale, zero_point = layer["quantization"]
        output_scales.append(scale)
        output_zero_points.append(zero_point)

    t0 = time.time()
    result = format_od_predictions(
        predictions,
        output_dtypes,
        output_scales,
        output_zero_points,
        score_threshold=0.05,
        img_shape=img.shape[1:3],
    )
    t1 = time.time()
    postprocessing_time = t1 - t0
    print(f"Postprocessing time: {postprocessing_time}")

    bboxes_label_names = []
    for bbox_label in result["bboxes_labels"]:
        bboxes_label_names.append(defect_map[int(bbox_label)])
    result["bboxes_label_names"] = bboxes_label_names
    return [result]


def format_od_predictions(
    prediction: list[np.ndarray],
    output_dtypes: list[np.dtype.type],
    output_scales: list[float],
    output_zero_points: list[int],
    score_threshold: float,
    img_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    pred_amount = len(prediction)
    assert pred_amount == 3, f"There are {pred_amount} predictions: {prediction}"

    idx = 0
    bboxes_scores = prediction[idx][0]
    bboxes_scores_output_dtype = output_dtypes[idx]
    if bboxes_scores_output_dtype == np.int8:
        bboxes_scores = denormalize_int8_scores(
            bboxes_scores, output_scales[idx], output_zero_points[idx]
        )

    idx = 1
    bboxes = prediction[idx][0]
    bboxes_output_dtype = output_dtypes[idx]
    if bboxes_output_dtype == np.int8:
        bboxes = denormalize_int8(bboxes, output_scales[idx], output_zero_points[idx])

    idx = 2
    bboxes_labels = prediction[idx][0]
    bboxes_labels_output_dtype = output_dtypes[idx]
    if bboxes_labels_output_dtype == np.int8:
        bboxes_labels = denormalize_int8_labels(
            bboxes_labels, output_scales[idx], output_zero_points[idx]
        )

    bboxes, bboxes_scores, bboxes_labels = sanitize_bboxes(
        (bboxes, bboxes_scores, bboxes_labels), score_threshold, img_shape
    )
    return {
        "bboxes_scores": bboxes_scores,
        "bboxes_labels": bboxes_labels,
        "bboxes": bboxes,
    }


def denormalize_int8(
    value: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    value = (value.astype(np.float32) - zero_point) * scale
    return value.astype(np.float32)


def format_dm(dm: dict[str, dict[str, str | int]]) -> dict[int, str]:
    return {int(k): v["name"] for k, v in dm.items()}


def unzip_bundle(bundle_path: Path) -> Path:
    with zipfile.ZipFile(bundle_path, "r") as zip_ref:
        unzipped_bundle_path = bundle_path.parent / bundle_path.stem
        zip_ref.extractall(unzipped_bundle_path)
        print(f"Unzipped bundle to {unzipped_bundle_path}")
    return unzipped_bundle_path


def filter_bboxes_out_of_bounds(
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray], img_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = img_shape
    bboxes, bboxes_scores, bboxes_labels = prediction
    x_min, y_min, x_max, y_max = bboxes.T
    in_bounds = np.where(
        (x_min >= 0)
        & (y_min >= 0)
        & (x_max <= width)
        & (y_max <= height)
        & (x_max > x_min)
        & (y_max > y_min)
    )[0]

    if in_bounds.shape[0] > 0:
        bboxes = bboxes[in_bounds]
        bboxes_scores = bboxes_scores[in_bounds]
        bboxes_labels = bboxes_labels[in_bounds]
    else:
        bboxes = np.array([]).reshape(-1, 4)
        bboxes_scores = np.array([])
        bboxes_labels = np.array([])

    return bboxes, bboxes_scores, bboxes_labels


def filter_bboxes_by_score_threshold(
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray], score_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function filters bboxes by a specified score threshold (exclusive).
    Namely, any bboxes whose score is <= score_threshold will be filtered out.
    """
    bboxes, bboxes_scores, bboxes_labels = prediction
    above_threshold = np.where(bboxes_scores > score_threshold)[0]

    if above_threshold.shape[0] > 0:
        bboxes = bboxes[above_threshold]
        bboxes_scores = bboxes_scores[above_threshold]
        bboxes_labels = bboxes_labels[above_threshold]
    else:
        bboxes = np.array([]).reshape(-1, 4)
        bboxes_scores = np.array([])
        bboxes_labels = np.array([])

    return bboxes, bboxes_scores, bboxes_labels


def sanitize_bboxes(
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray],
    score_threshold: float,
    img_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes, bboxes_scores, bboxes_labels = prediction

    bboxes, bboxes_scores, bboxes_labels = filter_bboxes_by_score_threshold(
        (bboxes, bboxes_scores, bboxes_labels),
        score_threshold,
    )
    bboxes, bboxes_scores, bboxes_labels = filter_bboxes_out_of_bounds(
        (bboxes, bboxes_scores, bboxes_labels), img_shape
    )
    return bboxes, bboxes_scores, bboxes_labels


def expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run(bundle_path: str, img_path: str, ext_delegate: str | None) -> None:
    # Load bundle
    bundle_path = Path(bundle_path)
    if bundle_path.suffix == ".zip":
        bundle_path = unzip_bundle(bundle_path)
    model_path = bundle_path / "model" / "saved_model.tflite"
    dm_path = bundle_path / "dm.json"
    with open(dm_path, "r") as source:
        defect_map = json.load(source)
    defect_map = format_dm(defect_map)

    # Load model
    interpreter, input_details, output_details = get_tflite_interpreter(
        str(model_path), ext_delegate=ext_delegate
    )

    # Load image and preprocess it
    img = read_img_rgb(img_path)
    img = preprocess(img, input_details)

    # Run inferences
    print("Running first inference")
    ret = run_inference(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        img=img,
        defect_map=defect_map,
    )
    print("Result:", ret)
    print("Running second inference")
    ret = run_inference(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        img=img,
        defect_map=defect_map,
    )
    print("Result:", ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", help="LandingLens bundle path")
    parser.add_argument("img_path", help="Path to image to run inference on")
    parser.add_argument("ext_delegate", nargs="?", help="Path to external delegate")
    args = parser.parse_args()
    run(args.bundle_path, args.img_path, args.ext_delegate)
