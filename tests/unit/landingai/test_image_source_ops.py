from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from landingai.image_source_ops import (
    probe_video,
    sample_images_from_video,
    take_photo_from_webcam,
)


def test_probe():
    test_video_file_path = "tests/data/videos/test.mp4"
    total_frames, sample_size, video_length_seconds = probe_video(
        test_video_file_path, 1.0
    )
    assert total_frames == 48
    assert sample_size == 2
    assert video_length_seconds == 2.0


def test_probe_file_not_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        non_exist_file = str(tmp_path / "non_exist.mp4")
        probe_video(non_exist_file, 1.0)


def test_sample_images_from_video(tmp_path: Path):
    test_video_file_path = "tests/data/videos/test.mp4"
    result = sample_images_from_video(test_video_file_path, tmp_path)
    assert len(result) == 2
    assert len(list(tmp_path.glob("*.jpg"))) == 2


@mock.patch("landingai.image_source_ops.cv2.imshow")
@mock.patch("landingai.image_source_ops.cv2.waitKey")
@mock.patch("landingai.image_source_ops.cv2.VideoCapture")
def test_take_photo_from_webcam(mocked_video_capture, mocked_wait_key, mocked_imshow):
    mocked_video_capture.return_value.read.return_value = (
        True,
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    mocked_wait_key.return_value = 32
    img = take_photo_from_webcam()
    assert img.size == (640, 480)
