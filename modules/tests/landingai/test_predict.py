import logging

import cv2
from landingai.predict import Predictor


def test_predict():
    api_key = ""
    api_secret = ""
    endpoint_id = ""
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = cv2.imread("modules/tests/data/cereal1.jpeg")
    assert img is not None
    res = predictor.predict(img)
    assert res, "Result should not be empty or None"
    logging.info(res)
