import logging
from pathlib import Path

import numpy as np
from PIL import Image

from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

api_key = "zm7ml657kh9a370k9liluxg9heuoufv"
api_secret = "1ccnesqy4em8dc32k2h2cu5kovdcd6palepaw4ugly6ttfl2fylu340x7ecja0"


def test_od_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/1/pr/21529989074947/deployment?project_purpose=regular&device=test1&tab=historical-data
    endpoint_id = "2d299622-434f-4ce9-b2eb-1142cdcfafcc"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = np.asarray(Image.open("tests/data/images/landing-logo.jpeg"))
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    res = predictor.predict(img)
    assert res, "Result should not be empty or None"
    logging.info(res)
    img_with_preds = overlay_predictions(predictions=res, image=img)
    img_with_preds.save("tests/output/test_od.jpg")


def test_seg_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/1/pr/10794/deployment?project_purpose=regular&device=asia_test&tab=historical-data
    endpoint_id = "3d2edb1b-073d-4853-87ca-30e430f84379"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = np.asarray(Image.open("tests/data/images/cereal1.jpeg"))
    assert img is not None
    preds = predictor.predict(img)
    assert preds, "Result should not be empty or None"
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_seg.jpg")


def test_vp_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/1/pr/22381224089609/deployment?project_purpose=regular&device=asia_test&tab=historical-data
    endpoint_id = "16049857-67bf-4c60-b20b-899741adbfdf"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = np.asarray(Image.open("tests/data/images/farm-coverage.jpg"))
    assert img is not None
    preds = predictor.predict(img)
    assert preds, "Result should not be empty or None"
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_vp.jpg")


def test_class_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/1/pr/22404917876739/deployment?project_purpose=regular&device=asia_test&tab=historical-data
    endpoint_id = "9f237028-e630-4576-8826-f35ab9003abe"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = np.asarray(Image.open("tests/data/images/wildfire1.jpeg"))
    assert img is not None
    preds = predictor.predict(img)
    assert preds, "Result should not be empty or None"
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_class.jpg")
