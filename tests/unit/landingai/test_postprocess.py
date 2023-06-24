import numpy as np
import responses
from PIL import Image

from landingai.postprocess import segmentation_class_pixel_coverage
from landingai.predict import Predictor


@responses.activate
def test_segmentation_class_pixel_coverage():
    responses._add_from_file(
        file_path="tests/data/responses/default_vp_model_response.yaml"
    )
    endpoint_id = "63035608-9d24-4342-8042-e4b08e084fde"
    predictor = Predictor(endpoint_id, "fake_key_12345", "fake_secret_12345")
    img = np.asarray(Image.open("tests/data/images/farm-coverage.jpg"))
    predictions = []
    for img in [img] * 3:
        predictions.extend(predictor.predict(img))
    coverage = segmentation_class_pixel_coverage(predictions)
    assert len(coverage) == 4
    assert coverage[3] == (0.007018192416185215, "Green Field")
    assert coverage[4] == (0.24161325195570196, "Brown Field")
    assert coverage[5] == (0.40297209139620843, "Trees")
    assert coverage[6] == (0.340975356067672, "Structure")
