import numpy as np
import pytest
import responses
from PIL import Image

from landingai.common import ClassificationPrediction, ObjectDetectionPrediction
from landingai.postprocess import (
    crop,
    rescale_bboxes,
    rescale_bboxes_by_image_size,
    segmentation_class_pixel_coverage,
)
from landingai.predict import Predictor


@responses.activate
def test_segmentation_class_pixel_coverage():
    responses._add_from_file(
        file_path="tests/data/responses/default_vp_model_response.yaml"
    )
    endpoint_id = "63035608-9d24-4342-8042-e4b08e084fde"
    predictor = Predictor(endpoint_id, api_key="land_sk_12345")
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


def test_rescale_bboxes_by_image_size():
    input = [
        ObjectDetectionPrediction(
            id="123",
            score=0.1,
            label_index=1,
            label_name="class1",
            bboxes=(4, 10, 22, 23),
        ),
        ObjectDetectionPrediction(
            id="124",
            score=0.1,
            label_index=1,
            label_name="class2",
            bboxes=(0, 0, 17, 33),
        ),
    ]
    raw_image = Image.new("RGB", (100, 200))
    resized_image = raw_image.resize((20, 60))
    result = rescale_bboxes_by_image_size(input, raw_image, resized_image)
    assert result[0].bboxes == (0, 3, 5, 7)
    assert result[1].bboxes == (0, 0, 4, 10)

    raw_image = Image.new("RGB", (150, 300))
    resized_image = raw_image.resize((100, 200))
    result = rescale_bboxes_by_image_size(input, resized_image, raw_image)
    assert result[0].bboxes == (6, 15, 33, 35)
    assert result[1].bboxes == (0, 0, 26, 50)


def test_rescale_bboxes():
    input = [
        ObjectDetectionPrediction(
            id="123",
            score=0.1,
            label_index=1,
            label_name="class1",
            bboxes=(4, 10, 22, 23),
        ),
        ObjectDetectionPrediction(
            id="124",
            score=0.1,
            label_index=1,
            label_name="class2",
            bboxes=(0, 0, 17, 33),
        ),
    ]
    result = rescale_bboxes(input, (0.3, 0.2))
    assert result[0].bboxes == (0, 3, 5, 7)
    assert result[1].bboxes == (0, 0, 4, 10)

    result = rescale_bboxes(input, 1.5)
    assert result[0].bboxes == (6, 15, 33, 35)
    assert result[1].bboxes == (0, 0, 26, 50)


def test_crop():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    img[20:31, 40:61, :] = 255
    img[40:51, 90:96, :] = 254
    preds = [
        ObjectDetectionPrediction(
            score=0.9, label_name="A", label_index=1, id="1", bboxes=(40, 20, 60, 30)
        ),
        ObjectDetectionPrediction(
            score=0.9, label_name="B", label_index=2, id="2", bboxes=(90, 40, 95, 50)
        ),
    ]
    output = crop(preds, img)
    assert output[0].size == (20, 10)
    assert np.count_nonzero(np.asarray(output[0])) == 20 * 10 * 3
    assert output[1].size == (5, 10)
    assert np.count_nonzero(np.asarray(output[1])) == 5 * 10 * 3
    # Empty preds should return empty list
    assert crop([], img) == []


def test_crop_with_invalid_prediction():
    prediction = ClassificationPrediction(
        id="123",
        label_index=1,
        label_name="class1",
        score=0.5,
    )
    img = np.zeros((100, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop([prediction], img)
