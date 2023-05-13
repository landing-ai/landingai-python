import logging
from pathlib import Path

import cv2

from landingai.predict import Predictor
from landingai.visualize import overlay_bboxes, overlay_colored_masks

api_key = "zm7ml657kh9a370k9liluxg9heuoufv"
api_secret = "1ccnesqy4em8dc32k2h2cu5kovdcd6palepaw4ugly6ttfl2fylu340x7ecja0"


def test_od_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    endpoint_id = "2d299622-434f-4ce9-b2eb-1142cdcfafcc"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = cv2.imread("tests/data/landing-logo.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    res = predictor.predict(img)
    assert res, "Result should not be empty or None"
    logging.info(res)
    img_with_preds = overlay_bboxes(predictions=res, image=img)
    img_with_preds.save("tests/output/test_od.jpg")


def test_seg_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    endpoint_id = "3d2edb1b-073d-4853-87ca-30e430f84379"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = cv2.imread("tests/data/cereal1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert preds, "Result should not be empty or None"
    logging.info(preds)
    img_with_masks = overlay_colored_masks(preds, img)
    img_with_masks.save("tests/output/test_seg.jpg")
