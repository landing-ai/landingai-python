import os
from pathlib import Path
from typing import List
from unittest import mock

import cv2
import numpy as np
import PIL.Image
import pytest

import landingai.pipeline as pl
from landingai.pipeline.frameset import Frame
from landingai.pipeline.image_source import (
    ImageFolder,
    NetworkedCamera,
    Screenshot,
    Webcam,
)


def test_image_folder_for_loop(input_image_folder):
    p = input_image_folder.glob("*")
    expected_files = [str(x) for x in p if x.is_file()]
    expected_files.sort()
    with ImageFolder(input_image_folder) as folder:
        assert len(list(folder)) == 8
        for i, frame in enumerate(folder):
            assert isinstance(frame, Frame)
            assert frame.image.size == (20, 20)
            assert frame.metadata["image_path"] == expected_files[i]
        assert len(list(folder)) == 8


def test_image_folder_with_input_files(input_image_files):
    with ImageFolder(input_image_files) as folder:
        assert len(list(folder)) == 8
        for i, frame in enumerate(folder):
            assert isinstance(frame, Frame)
            assert frame.image.size == (20, 20)
            assert frame.metadata["image_path"] == input_image_files[i]


def test_image_folder_with_glob_patterns(input_image_folder):
    with input_image_folder as folder:
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


def test_networked_camera():
    # Use a video to simulate a live camera and test motion detection
    test_video_file_path = "tests/data/videos/countdown.mp4"
    with NetworkedCamera(
        stream_url=test_video_file_path, motion_detection_threshold=50
    ) as camera:
        # Get the first frame and the next frame where motion is detected. I keep the detection threshold low to make the test fast
        i = iter(camera)
        frame1 = next(i)
        while True:
            # if we cannot get any motion detection (e.g. Threshold 100%), next() will throw an exception and fail the test
            frame2 = next(i)
            if frame2 is not None:
                break
        # frame1.show_image()
        # frame2.show_image()
        image_distance = np.sum(
            cv2.absdiff(src1=frame1.to_numpy_array(), src2=frame2.to_numpy_array())
        )

        # Compute the diff by summing the delta between each pixel across the two images
        assert (
            image_distance > 100000
        )  # Even with little motion this number should exceed 100k


@mock.patch("landingai.pipeline.image_source.cv2")
def test_webcam(mock_cv2):
    """Makes sure webcam is a networked camare that just uses camera index as stream URL"""
    with Webcam() as cam:
        assert isinstance(cam, NetworkedCamera)
        mock_cv2.VideoCapture.assert_called_once_with(0)


@mock.patch(
    "landingai.pipeline.image_source.ImageGrab.grab",
    return_value=PIL.Image.open("tests/data/images/cereal1.jpeg"),
)
def test_screenshot(mock_img_grab):
    with Screenshot() as screenshot:
        frame = next(screenshot)
        assert isinstance(frame, Frame)
        expected_img = PIL.Image.open("tests/data/images/cereal1.jpeg")
        assert (np.asarray(frame.image) == np.asarray(expected_img)).all()


def test_videofile_properties():
    video_file = "tests/data/videos/test.mp4"
    with pl.image_source.VideoFile(video_file, samples_per_second=8) as video_source:
        (s_fps, s_total_frames, t_fps, t_total_frames) = video_source.properties()
        assert s_total_frames == 48
        assert s_fps == 24
        assert t_total_frames == 16
