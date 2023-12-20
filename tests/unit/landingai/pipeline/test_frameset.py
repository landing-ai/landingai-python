import os
from pathlib import Path
from unittest import mock

from numpy.testing import assert_array_equal
from PIL import Image
import pytest

from landingai.common import (
    ObjectDetectionPrediction,
    OcrPrediction,
    ClassificationPrediction,
)
from landingai.storage.data_access import fetch_from_uri
from landingai.pipeline.frameset import FrameSet, Frame, PredictionList


def get_frameset(image_path: str = "tests/data/images/cereal1.jpeg") -> FrameSet:
    return FrameSet.from_image(image_path)


def get_frame(image_path: str = "tests/data/images/cereal1.jpeg") -> Frame:
    return Frame.from_image(image_path)


def get_frameset_with_od_coffee_prediction(
    image_path: str = "tests/data/images/cereal1.jpeg",
) -> FrameSet:
    frameset = get_frameset(image_path)
    frameset.frames[0].predictions = PredictionList(
        [
            ObjectDetectionPrediction(
                score=0.6,
                label_name="coffee",
                label_index=1,
                id="aaaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                bboxes=(352, 147, 463, 248),
            )
        ]
    )
    return frameset


def get_frame_with_od_coffee_prediction(
    image_path: str = "tests/data/images/cereal1.jpeg",
) -> Frame:
    frame = get_frame(image_path)
    frame.predictions = PredictionList(
        [
            ObjectDetectionPrediction(
                score=0.6,
                label_name="coffee",
                label_index=1,
                id="aaaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                bboxes=(352, 147, 463, 248),
            )
        ]
    )
    return frame


def get_frame_with_ocr_predictions() -> Frame:
    frame = get_frame()
    frame.predictions = PredictionList(
        [
            OcrPrediction(
                score=0.6,
                text="there is some text",
                location=[(19, 23), (289, 45), (284, 113), (14, 91)],
            ),
            OcrPrediction(
                score=0.7,
                text="in this image",
                location=[(13, 105), (308, 108), (308, 190), (12, 188)],
            ),
        ]
    )
    return frame


def get_frameset_with_ocr_predictions() -> Frame:
    frame = get_frameset()
    frame.frames[0].predictions = PredictionList(
        [
            OcrPrediction(
                score=0.6,
                text="there is some text",
                location=[(19, 23), (289, 45), (284, 113), (14, 91)],
            ),
            OcrPrediction(
                score=0.7,
                text="in this image",
                location=[(13, 105), (308, 108), (308, 190), (12, 188)],
            ),
        ]
    )
    return frame


def test_save_video(tmp_path: Path):
    frs = FrameSet.from_image("tests/data/images/cereal1.jpeg")

    # Test extension validation
    with pytest.raises(NotImplementedError):
        frs.save_video(str(tmp_path / "video.mp3"))

    filename = str(tmp_path / "video.mp4")
    # Test the combinatorics of fps & video length
    with pytest.raises(ValueError):
        frs.save_video(filename, video_length_sec=2, video_fps=1)

    # Create a 1 frame video with default params
    frs.save_video(filename)
    assert os.path.exists(filename)


def test_predictions_empty():
    frameset = FrameSet.from_image("tests/data/images/cereal1.jpeg")
    assert isinstance(frameset.predictions, list)
    assert isinstance(frameset.predictions, PredictionList)
    assert len(frameset.predictions) == 0


def test_predictions_with_extra_frames_in_frameset():
    frameset_with_od_coffee_prediction = get_frameset_with_od_coffee_prediction()
    img = Image.open(str(fetch_from_uri("tests/data/images/cereal1.jpeg")))

    other_prediction = ObjectDetectionPrediction(
        score=0.8,
        label_name="coffee",
        label_index=1,
        id="bbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbbb",
        bboxes=(352, 147, 463, 248),
    )
    frame = Frame(image=img, metadata={})
    frame.predictions = [other_prediction]
    frameset_with_od_coffee_prediction.frames.append(frame)

    # Make sure frameset.predictions is a PredictionList that inherits from list
    assert isinstance(frameset_with_od_coffee_prediction.predictions, list)
    assert isinstance(frameset_with_od_coffee_prediction.predictions, PredictionList)

    assert (
        len(frameset_with_od_coffee_prediction.predictions) == 2
    ), "frameset.predictions should concatenate predictions from all frames"
    assert frameset_with_od_coffee_prediction.predictions == [
        frameset_with_od_coffee_prediction.frames[0].predictions[0],
        frameset_with_od_coffee_prediction.frames[1].predictions[0],
    ]


def test_od_predictions_contains_label():
    frameset_with_od_coffee_prediction = get_frameset_with_od_coffee_prediction()
    assert "coffee" in frameset_with_od_coffee_prediction.predictions
    assert "tea" not in frameset_with_od_coffee_prediction.predictions


def test_od_predictions_filter_threshold():
    frameset_with_od_coffee_prediction = get_frameset_with_od_coffee_prediction()
    assert "coffee" in frameset_with_od_coffee_prediction.predictions
    assert "tea" not in frameset_with_od_coffee_prediction.predictions

    # "coffee" has a score of 0.6, so this call should filter it out
    filtered_preds = frameset_with_od_coffee_prediction.predictions.filter_threshold(
        0.7
    )
    assert "coffee" not in filtered_preds
    assert "tea" not in filtered_preds


def test_od_predictions_filter_label():
    frameset_with_od_coffee_prediction = get_frameset_with_od_coffee_prediction()
    assert "coffee" in frameset_with_od_coffee_prediction.predictions
    assert "tea" not in frameset_with_od_coffee_prediction.predictions

    # filter for "coffee" label should keep everything the same
    filtered_preds = frameset_with_od_coffee_prediction.predictions.filter_label(
        "coffee"
    )
    assert "coffee" in filtered_preds
    assert "tea" not in filtered_preds

    # filter for "tea" label should remove "coffee" prediction
    filtered_preds = frameset_with_od_coffee_prediction.predictions.filter_label("tea")
    assert "coffee" not in filtered_preds
    assert "tea" not in filtered_preds


def test_ocr_predictions_crop_predictions():
    frame = get_frame()
    frame.predictions = PredictionList(
        [
            OcrPrediction(
                score=0.6,
                text="there is some text",
                # This should yield bounding box (65, 19, 133, 33); shape (68, 14)
                location=[(65, 19), (133, 19), (133, 33), (65, 33)],
            ),
            OcrPrediction(
                score=0.7,
                text="in this image",
                # The location for OCR might come a bit inclined. The returned bbox should account for that.
                # This should yield bounding box (80, 54, 136, 72); shape (56, 18)
                location=[(81, 54), (136, 58), (135, 72), (80, 69)],
            ),
        ]
    )
    pred_frames = frame.crop_predictions()
    assert isinstance(pred_frames, FrameSet)
    assert len(pred_frames) == 2
    assert pred_frames[0].predictions == [frame.predictions[0]]
    assert pred_frames[0].image.size == (68, 14)

    assert pred_frames[1].predictions == [frame.predictions[1]]
    assert pred_frames[1].image.size == (56, 18)


def test_od_predictions_crop_predictions():
    frame = get_frame()
    frame.predictions = PredictionList(
        [
            ObjectDetectionPrediction(
                score=0.8,
                label_name="coffee",
                label_index=1,
                id="bbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbbb",
                # Shape: (111, 101)
                bboxes=(352, 147, 463, 248),
            ),
            ObjectDetectionPrediction(
                score=0.8,
                label_name="coffee",
                label_index=1,
                id="bbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbbb",
                # Shape: (66, 221)
                bboxes=(33, 1, 99, 222),
            ),
        ]
    )

    pred_frames = frame.crop_predictions()
    assert isinstance(pred_frames, FrameSet)
    assert len(pred_frames) == 2
    assert pred_frames[0].predictions == [frame.predictions[0]]
    assert pred_frames[0].image.size == (111, 101)

    assert pred_frames[1].predictions == [frame.predictions[1]]
    assert pred_frames[1].image.size == (66, 221)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_frame_dot_frames_returns_all_available_frames(frame_getter):
    # TODO: This test should be deprecated together with Frame.frames property
    frame = frame_getter()
    assert isinstance(frame.frames, list)
    assert len(frame.frames) == 1
    assert isinstance(frame.frames[0].image.size, tuple)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_resize(frame_getter):
    frame = frame_getter()
    frame.resize(width=100, height=100)
    assert get_image_from_frame_or_frameset(frame).size == (100, 100)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_frameset_downsize(frame_getter):
    frame = frame_getter()
    frame.downsize(width=5, height=5)
    assert get_image_from_frame_or_frameset(frame).size == (5, 5)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_frameset_downsize_bigger_than_original(frame_getter):
    # "Downsizing" the image to a bigger size should not change the image size
    frame = frame_getter()
    image = get_image_from_frame_or_frameset(frame)
    width, height = image.size
    frame.downsize(width=width + 10, height=height + 10)
    assert get_image_from_frame_or_frameset(frame).size == (width, height)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_crop(frame_getter):
    frame = frame_getter()
    image = get_image_from_frame_or_frameset(frame)
    width, height = image.size
    # Crop 5px from each side
    frame.crop((5, 5, width - 5, height - 5))
    assert get_image_from_frame_or_frameset(frame).size == (width - 10, height - 10)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
@pytest.mark.parametrize(
    "enhancement",
    [
        ("adjust_sharpness", 1.0),
        ("adjust_brightness", 1.0),
        ("adjust_contrast", 1.0),
        ("adjust_color", 1.0),
    ],
)
def test_enhancements_no_op(enhancement, frame_getter):
    """Checks each enhancement method with it's no-op factor, making sure the frame
    will be exactly the same after the operation"""
    enhancement_method, enhancement_factor = enhancement
    frame = frame_getter()
    original_frame = frame.copy()
    getattr(frame, enhancement_method)(enhancement_factor)

    original_image = get_image_from_frame_or_frameset(original_frame)
    image = get_image_from_frame_or_frameset(frame)
    assert original_image is not image
    assert_array_equal(original_image, image)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
@pytest.mark.parametrize(
    "enhancement",
    [
        ("adjust_sharpness", 1.5, "sharpness-1.5.jpeg"),
        ("adjust_brightness", 1.5, "brightness-1.5.jpeg"),
        ("adjust_contrast", 1.5, "contrast-1.5.jpeg"),
        ("adjust_color", 1.5, "color-1.5.jpeg"),
    ],
)
def test_enhancements(enhancement, frame_getter):
    """Checks each enhancement method with it's factor, making sure the frame
    wil be changed after the operation"""
    enhancement_method, enhancement_factor, expected_img_file = enhancement
    img_folder = "tests/data/images/cereal-tiny/"
    frame = frame_getter(image_path=f"{img_folder}/original.jpeg")
    original_frame = frame.copy()
    getattr(frame, enhancement_method)(enhancement_factor)

    original_image = get_image_from_frame_or_frameset(original_frame)
    image = get_image_from_frame_or_frameset(frame)
    assert original_image is not image

    expected_content = Image.open(f"{img_folder}/{expected_img_file}")
    assert_array_equal(image, expected_content)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_od_coffee_prediction,
        get_frame_with_od_coffee_prediction,
    ],
)
def test_copy_operation(frame_getter):
    frame = frame_getter()
    width, height = get_image_from_frame_or_frameset(frame).size
    new_frame = frame.copy()
    # Do any operation on the new frame, so we can test if the original frame is preserved
    new_frame.crop((0, 0, 1, 1))

    original_image = get_image_from_frame_or_frameset(frame)
    new_image = get_image_from_frame_or_frameset(new_frame)
    assert frame.predictions == new_frame.predictions
    assert original_image is not new_image
    assert original_image.size == (width, height)


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset,
        get_frame,
    ],
)
def test_predict_uses_raw_pil_image(frame_getter):
    """Test that the predict method uses the raw PIL image, not the array-converted image.

    This is to ensure that the image preserves its particularities when sent to predictor.
    When sending the array-converted image, RGBA images will raise convertion errors, for example.
    """
    frame = frame_getter()
    predictor = mock.Mock()
    predictor.predict.return_value = []
    frame.run_predict(predictor)
    assert predictor.predict.call_count == 1
    assert predictor.predict.call_args[0][0] is get_image_from_frame_or_frameset(frame)


def test_run_predict_forwards_kwargs():
    """Test that the predict method uses the raw PIL image, not the array-converted image.

    This is to ensure that the image preserves its particularities when sent to predictor.
    When sending the array-converted image, RGBA images will raise convertion errors, for example.
    """
    frame = get_frame()
    predictor = mock.Mock()
    predictor.predict.return_value = []
    frame.run_predict(predictor, k1="a", k2="b", k3=3)
    predictor.predict.assert_called_once_with(
        get_image_from_frame_or_frameset(frame),
        reuse_session=True,
        k1="a",
        k2="b",
        k3=3,
    )


@pytest.mark.parametrize(
    "frame_getter",
    [
        get_frameset_with_ocr_predictions,
        get_frame_with_ocr_predictions,
    ],
)
def test_ocr_predictions_contains_message(frame_getter):
    frame = frame_getter()
    assert "some text" in frame.predictions
    assert "some text in" in frame.predictions
    assert "not-present text" not in frame.predictions


def get_image_from_frame_or_frameset(frame_or_frameset) -> Image:
    if isinstance(frame_or_frameset, FrameSet):
        return frame_or_frameset.frames[0].image
    else:
        return frame_or_frameset.image


def test_prediction_list_only_has_one_prediction_type():
    with pytest.raises(ValueError):
        PredictionList(
            [
                ClassificationPrediction(score=1.0, label_name="house", label_index=1),
                OcrPrediction(score=1.0, text="LandingAI", location=[]),
            ]
        )
