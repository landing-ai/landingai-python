import io
from pathlib import Path

import responses
from PIL import Image
from responses.matchers import multipart_matcher

from landingai.predict import Predictor


@responses.activate
def test_predict_matching_expected_request_body():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    img_path = "tests/data/images/wildfire1.jpeg"
    img = Image.open(img_path)
    expected_request_files = {
        "file": ("image.png", _read_image_as_png_bytes(img_path), "image/png")
    }
    responses.post(
        url="https://predict.app.landing.ai/inference/v1/predict?endpoint_id=8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43",
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


def _read_image_as_png_bytes(file_path: str) -> bytes:
    img_buffer = io.BytesIO()
    image = Image.open(file_path)
    image.save(img_buffer, format="PNG")
    data = img_buffer.getvalue()
    assert data is not None
    return data
