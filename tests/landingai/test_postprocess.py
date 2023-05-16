import numpy as np
from PIL import Image

from landingai.postprocess import segmentation_class_pixel_coverage
from landingai.predict import Predictor

api_key = "zm7ml657kh9a370k9liluxg9heuoufv"
api_secret = "1ccnesqy4em8dc32k2h2cu5kovdcd6palepaw4ugly6ttfl2fylu340x7ecja0"


def test_segmentation_class_pixel_coverage():
    endpoint_id = "16049857-67bf-4c60-b20b-899741adbfdf"
    predictor = Predictor(endpoint_id, api_key, api_secret)
    img = np.asarray(Image.open("tests/data/farm-coverage.jpg"))
    predictions = []
    for img in [img] * 3:
        predictions.extend(predictor.predict(img))
    coverage = segmentation_class_pixel_coverage(predictions)
    assert len(coverage) == 4
    assert coverage[1] == (0.034910176946490436, "green field")
    assert coverage[2] == (0.09148953979644242, "brown field")
    assert coverage[3] == (0.024819751951445816, "structure")
    assert coverage[4] == (0.09751108322318006, "trees")
