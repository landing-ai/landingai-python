import cv2
import numpy as np
import os
from pathlib import Path

import pytest

from landingai.vision_pipeline import NetworkedCamera, FrameSet
from landingai.common import ObjectDetectionPrediction


def test_networked_camera():
    # Use a video to simulate a live camera and test motion detection
    test_video_file_path = "tests/data/videos/countdown.mp4"
    Camera = NetworkedCamera(
        stream_url=test_video_file_path, motion_detection_threshold=50
    )

    # Get the first frame and the next frame where motion is detected. I keep the detection threshold low to make the test fast
    i = iter(Camera)
    frame1 = next(i)
    while True:
        # if we cannot get any motion detection (e.g. Threshold 100%), next() will throw an exception and fail the test
        frame2 = next(i)
        if not frame2.is_empty():
            break
    # frame1.show_image()
    # frame2.show_image()
    image_distance = np.sum(
        cv2.absdiff(src1=frame1[0].to_numpy_array(), src2=frame2[0].to_numpy_array())
    )

    # Compute the diff by summing the delta between each pixel across the two images
    del Camera
    assert (
        image_distance > 100000
    )  # Even with little motion this number should exceed 100k


def test_class_counts():
    preds = [
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

    frs = FrameSet.from_image("tests/data/images/cereal1.jpeg")
    frs[0].predictions = preds
    counts = frs.get_class_counts()
    assert counts["screw"] == 3


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
