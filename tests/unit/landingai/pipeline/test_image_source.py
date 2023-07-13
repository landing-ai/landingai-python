import os
from pathlib import Path
from typing import List

import PIL.Image
import cv2
import numpy as np
import pytest

from landingai.pipeline.image_source import (
    ImageFolder,
    NetworkedCamera,
    probe_video,
    sample_images_from_video,
)
from landingai.pipeline.frameset import FrameSet


def test_image_folder_for_loop(input_image_folder):
    p = input_image_folder.glob("*")
    expected_files = [str(x) for x in p if x.is_file()]
    expected_files.sort()
    folder = ImageFolder(input_image_folder)
    assert len(list(folder)) == 8
    for i, frames in enumerate(folder):
        assert isinstance(frames, FrameSet)
        assert frames[0].image.size == (20, 20)
        assert frames[0].metadata["image_path"] == expected_files[i]
    assert len(list(folder)) == 8


def test_image_folder_with_input_files(input_image_files):
    folder = ImageFolder(input_image_files)
    assert len(list(folder)) == 8
    for i, frames in enumerate(folder):
        assert isinstance(frames, FrameSet)
        assert frames[0].image.size == (20, 20)
        assert frames[0].metadata["image_path"] == input_image_files[i]


def test_image_folder_with_glob_patterns(input_image_folder):
    folder = input_image_folder
    jpg_cnt = 2 if os.name == "nt" else 1
    # Test __len__
    assert len(ImageFolder(glob_pattern=[str(folder / "*.jpg")])) == jpg_cnt
    assert len(ImageFolder(glob_pattern=str(folder / "*.bmp"))) == 1
    # Test __iter__
    assert len(list(ImageFolder(glob_pattern=[str(folder / "*.jpg")]))) == jpg_cnt
    assert len(list(ImageFolder(str(folder)))) == 8
    # Test glob pattern on nested files
    assert len(ImageFolder(glob_pattern=str(folder / "**/*.png"))) == 4


@pytest.fixture
def input_image_files(tmp_path_factory) -> List[Path]:
    tmp_dir = tmp_path_factory.mktemp("image_folder")
    file_names = [
        "test1.jpg",
        "test2.JPG",
        "test3.jpeg",
        "test4.png",
        "test5.jpx",
        "test6.tiff",
        "test7.bmp",
        "test8.gif",
    ]
    file_paths = []
    for name in file_names:
        file_path = tmp_dir / name
        PIL.Image.new(mode="RGB", size=(20, 20)).save(file_path)
        file_paths.append(file_path)
    # Create a sub-directory with nested image files and non-image files
    sub_dir = _create_file_under(tmp_dir, "sub_dir", "test9.png")
    _create_file_under(sub_dir, "double-nested", "test10.png")
    _create_file_under(sub_dir, None, "test.txt")
    _create_file_under(tmp_dir, "sub_dir2", "test11.png")
    return file_paths


def _create_file_under(tmp_dir, sub_dir, name) -> Path:
    if sub_dir is not None:
        sub_dir = tmp_dir / sub_dir
    else:
        sub_dir = tmp_dir
    sub_dir.mkdir(exist_ok=True, parents=True)
    if name.endswith(".png"):
        PIL.Image.new(mode="RGB", size=(10, 10)).save(sub_dir / name)
    elif name.endswith(".txt"):
        (sub_dir / name).touch()
    else:
        raise ValueError("Unsupported file type")
    return sub_dir


@pytest.fixture
def input_image_folder(input_image_files) -> Path:
    return input_image_files[0].parent


# TODO: solve the problem of exit code 134 when running the following test in GitHub Actions
# @patch("landingai.io.cv2.waitKey")
# @patch("landingai.io.cv2.VideoCapture")
# def test_read_from_notebook_webcam(mock_video_capture, mock_wait_key):
#     mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3)))
#     mock_wait_key.return_value = 288
#     take_photo_func = read_from_notebook_webcam()
#     filepath = take_photo_func()
#     image = PIL.Image.open(filepath)
#     assert image.size == (640, 480)


def test_sample_images_from_video(tmp_path: Path):
    test_video_file_path = "tests/data/videos/test.mp4"
    result = sample_images_from_video(test_video_file_path, tmp_path)
    assert len(result) == 2
    assert len(list(tmp_path.glob("*.jpg"))) == 2


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
