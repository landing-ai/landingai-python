from pathlib import Path

import cv2
import numpy as np
import pytest

from landingai.io import sample_images_from_video, probe_video


def test_sample_images_from_video(test_video_file_path: str, tmp_path: Path):
    result = sample_images_from_video(test_video_file_path, tmp_path)
    assert len(result) == 2
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_probe(test_video_file_path):
    total_frames, sample_size, video_length_seconds = probe_video(test_video_file_path, 1.0)
    assert total_frames == 48
    assert sample_size == 2
    assert video_length_seconds == 2.0

def test_probe_file_not_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        non_exist_file = str(tmp_path / "non_exist.mp4")
        probe_video(non_exist_file, 1.0)


@pytest.fixture
def test_video_file_path(tmp_path_factory) -> str:
    tmp_dir = tmp_path_factory.mktemp("video_output")
    video_file = str(tmp_dir / "test.mp4")
    sampe_frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"H264"), 24, (128, 128))
    total_frames = 48
    for _ in range(total_frames):
        video.write(sampe_frame)
    return video_file
