import logging
import io
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pytest
import responses
from PIL import Image
from responses.matchers import multipart_matcher

from landingai.exceptions import (
    BadRequestError,
    ClientError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitExceededError,
    ServiceUnavailableError,
    UnauthorizedError,
    UnexpectedRedirectError,
)
from landingai.predict import Predictor, EdgePredictor
from landingai.visualize import overlay_predictions


def test_predict_with_none():
    with pytest.raises(ValueError) as excinfo:
        Predictor("12345", "key", "secret").predict(None)
    assert "Input image must be non-emtpy, but got: None" in str(excinfo.value)


def test_predict_with_empty_array():
    with pytest.raises(ValueError) as excinfo:
        Predictor("12345", "key", "secret").predict(np.array([]))
    assert "Input image must be non-emtpy, but got: []" in str(excinfo.value)


@responses.activate
def test_predict_500():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_500.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(InternalServerError) as excinfo:
        Predictor("db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", "key", "secret").predict(img)
    assert (
        "Internal server error. The model server encountered an unexpected condition and failed. Please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_504():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_504.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ServiceUnavailableError) as excinfo:
        Predictor("db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", "key", "secret").predict(img)
    assert "Service temporarily unavailable. Please try again in a few minutes." in str(
        excinfo.value
    )


@responses.activate
def test_predict_503():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_503.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ServiceUnavailableError) as excinfo:
        Predictor("db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", "key", "secret").predict(img)
    assert "Service temporarily unavailable. Please try again in a few minutes." in str(
        excinfo.value
    )


@responses.activate
def test_predict_404():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_404.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(BadRequestError) as excinfo:
        with patch.object(Predictor, "_url", "https://predict.app.landing.ai/v0/foo"):
            Predictor("8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", "key", "secret").predict(
                img
            )
    assert (
        "Endpoint doesn't exist. Please check the inference url path and other configuration is correct and try again. If this issue persists, please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_400():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_400.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(ClientError) as excinfo:
        Predictor("db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4", "key", "secret").predict(img)
    assert (
        "Client error. Please check your configuration and inference request is well-formed and try again. If this issue persists, please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_300():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_300.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(UnexpectedRedirectError) as excinfo:
        Predictor("8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", "key", "secret").predict(img)
    assert (
        "Unexpected redirect. Please report this issue to LandingLens support for further assistant."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_429_rate_limited():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_429.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(RateLimitExceededError) as excinfo:
        Predictor._num_retry = 1
        Predictor("8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", "key", "secret").predict(img)
    assert (
        "Rate limit exceeded. You have sent too many requests in a minute. Please wait for a minute before sending new requests. Contact your account admin or LandingLens support for how to increase your rate limit."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_403_run_out_of_credits():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_403.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(PermissionDeniedError) as excinfo:
        Predictor(
            "dfa79692-75eb-4a48-b02e-b273751adbae",
            "1r1c7i0c1pnmp1oac1748fnrnerh7dg",
            "mreiix1wmm8da3qnzfe7xivwq53rsajnl7k8re0iz14zmm1s6gepdien82r871",
        ).predict(img)
    assert (
        "Permission denied. Please check your account has enough credits or your enterprise contract is not expired or if you have access to this endpoint. Contact your account admin for more information."
        in str(excinfo.value)
    )


@responses.activate
def test_predict_401_with_wrong_api_key():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_401.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(UnauthorizedError) as excinfo:
        Predictor(
            "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", "wrong_key", "wrong_secret"
        ).predict(img)
    assert "Unauthorized. Please check your API key and API secret is correct." in str(
        excinfo.value
    )


@responses.activate
def test_predict_422_with_wrong_endpoint_id():
    responses._add_from_file(
        file_path="tests/data/responses/v1_predict_status_422.yaml"
    )

    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    with pytest.raises(BadRequestError) as excinfo:
        Predictor("12345", "correct_api_key", "correct_secret").predict(img)
    assert "Bad request. Please check your Endpoint ID is correct." in str(
        excinfo.value
    )


@responses.activate
def test_predict_matching_expected_request_body():
    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    expected_request_files = {
        "file": ("image.png", _read_image_as_png_bytes(img_path), "image/png")
    }
    responses.post(
        url="https://predict.app.landing.ai/inference/v1/predict?endpoint_id=8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43&device_type=pylib",
        match=[multipart_matcher(expected_request_files)],
        json={
            "backbonetype": None,
            "backbonepredictions": None,
            "predictions": {
                "score": 0.9951885938644409,
                "labelIndex": 0,
                "labelName": "Fire",
            },
            "type": "ClassificationPrediction",
        },
    )
    predictor = Predictor(
        "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43", "fake_key", "fake_secret"
    )
    predictor.predict(img)


@responses.activate
def test_edge_class_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_class_predict.yaml"
    )
    # Project: https://app.landing.ai/app/376/pr/26119078438913/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 26119078438913 \
    #                                            -m "59eff733-1dcd-4ace-b104-9041c745f1da" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    img = Image.open("tests/data/images/wildfire1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    assert preds[0].label_name == "HasFire"
    assert preds[0].label_index == 0
    np.testing.assert_almost_equal(preds[0].score, 0.99565023, decimal=3, err_msg='class score mismatch')
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_edge_class.jpg")


@responses.activate
def test_edge_od_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Endpoint: https://app.landing.ai/app/376/pr/11165/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 11165 \
    #                                    -m "5d0b04ce-8327-465c-b270-6913d92f5936" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_od_predict.yaml"
    )
    img = np.asarray(Image.open("tests/data/images/cereal1.jpeg"))
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    preds = predictor.predict(img)
    assert len(preds) == 3, "Result should not be empty or None"
    expected_scores = [0.993014, 0.992160, 0.954738]
    expected_bboxes = [
        (945, 1603, 1118, 1795),
        (436, 1037, 640, 1203),
        (1515, 1419, 1977, 1787),
    ]
    for i, pred in enumerate(preds):
        assert pred.label_name == "Screw"
        assert pred.label_index == 1
        np.testing.assert_almost_equal(pred.score, expected_scores[i], decimal=3, err_msg='OD score mismatch')
        assert pred.bboxes == expected_bboxes[i]
    logging.info(preds)
    img_with_preds = overlay_predictions(predictions=preds, image=img)
    img_with_preds.save("tests/output/test_edge_od.jpg")


@responses.activate
def test_edge_seg_predict(expected_seg_prediction, seg_mask_validator):
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    responses._add_from_file(
        file_path="tests/data/responses/test_edge_seg_predict.yaml"
    )
    # Project: https://app.landing.ai/app/376/pr/26113016987660/deployment?device=tiger-team-integration-tests
    # run LandingEdge.CLI with cmdline parameters: run-online -k "your_api_key" -s "your_secret_key" -r 26113016987660 \
    #                                          -m "1a2b49dd-25c0-45fa-9f1c-4433acee80cd" -n test_edge_cli --port 8123
    predictor = EdgePredictor("localhost", 8123)
    img = Image.open("tests/data/images/cereal1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    seg_mask_validator(preds[0], expected_seg_prediction)
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_edge_seg.jpg")


def _read_image_as_png_bytes(file_path: str) -> bytes:
    img_buffer = io.BytesIO()
    image = Image.open(file_path)
    image.save(img_buffer, format="PNG")
    data = img_buffer.getvalue()
    assert data is not None
    return data
