import os
from pathlib import Path

import pytest

from landingai.pipeline.frameset import FrameSet


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
