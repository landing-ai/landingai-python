from landingai.common import ObjectDetectionPrediction
from landingai.pipeline.frameset import FrameSet, PredictionList
from landingai.pipeline.postprocessing import get_class_counts


def test_class_counts():
    preds = PredictionList(
        [
            ObjectDetectionPrediction(
                id="1",
                label_index=0,
                label_name="screw",
                score=0.623112,
                bboxes=(432, 1035, 651, 1203),
            ),
            ObjectDetectionPrediction(
                id="2",
                label_index=0,
                label_name="screw",
                score=0.892,
                bboxes=(1519, 1414, 1993, 1800),
            ),
            ObjectDetectionPrediction(
                id="3",
                label_index=0,
                label_name="screw",
                score=0.7,
                bboxes=(948, 1592, 1121, 1797),
            ),
        ]
    )

    frs = FrameSet.from_image("tests/data/images/cereal1.jpeg")
    frs[0].predictions = preds
    counts = get_class_counts(frs)
    assert counts["screw"] == 3
