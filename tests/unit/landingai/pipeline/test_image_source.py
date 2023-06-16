import PIL.Image
import pytest

from landingai.pipeline.image_source import ImageFolder
from landingai.vision_pipeline import FrameSet


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
    # Test __len__
    assert (
        len(
            ImageFolder(
                glob_pattern=[
                    str(input_image_folder / "*.jpg"),
                    str(input_image_folder / "*.JPG"),
                ]
            )
        )
        == 2
    )
    assert len(ImageFolder(glob_pattern=str(input_image_folder / "*.bmp"))) == 1
    # Test __iter__
    assert (
        len(
            list(
                ImageFolder(
                    glob_pattern=[
                        str(input_image_folder / "*.png"),
                        str(input_image_folder / "*.PNG"),
                    ]
                )
            )
        )
        == 2
    )
    assert len(list(ImageFolder(str(input_image_folder)))) == 8
    # Test glob pattern on nested files
    assert len(ImageFolder(glob_pattern=str(input_image_folder / "**/*.jpg"))) == 4


@pytest.fixture
def input_image_files(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("image_folder")
    file_names = [
        "test1.jpg",
        "test2.JPG",
        "test3.jpeg",
        "test4.png",
        "test5.PNG",
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
    sub_dir = _create_file_under(tmp_dir, "sub_dir", "test9.jpg")
    _create_file_under(sub_dir, "double-nested", "test10.jpg")
    _create_file_under(sub_dir, None, "test.txt")
    _create_file_under(tmp_dir, "sub_dir2", "test11.jpg")
    return file_paths


def _create_file_under(tmp_dir, sub_dir, name):
    if sub_dir is not None:
        sub_dir = tmp_dir / sub_dir
    else:
        sub_dir = tmp_dir
    sub_dir.mkdir(exist_ok=True, parents=True)
    if name.endswith(".jpg"):
        PIL.Image.new(mode="RGB", size=(10, 10)).save(sub_dir / name)
    elif name.endswith(".txt"):
        (sub_dir / name).touch()
    else:
        raise ValueError("Unsupported file type")
    return sub_dir


@pytest.fixture
def input_image_folder(input_image_files):
    return input_image_files[0].parent
