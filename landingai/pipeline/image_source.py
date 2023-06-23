"""A module that provides a set of abstractions and APIs for reading images from different sources."""

import glob
from collections.abc import Iterator
from pathlib import Path
from typing import Iterator as IteratorType
from typing import List, Union

from landingai.vision_pipeline import FrameSet


class ImageSourceBase(Iterator):
    """The base class for all image sources."""

    def __next__(self) -> FrameSet:
        raise NotImplementedError()


class ImageFolder(ImageSourceBase):
    """
    The `ImageFolder` class is an image source that reads images from a folder path.

    Example 1:
    ```
    folder = ImageFolder("/home/user/images")
    for image_batch in folder:
        print(image_batch[0].image.size)
    ```

    Example 2:
    ```
    # Read all jpg files in the folder (including nested files)
    folder = ImageFolder(glob_pattern="/home/user/images/**/*.jpg")
    for image_batch in folder:
        print(image_batch[0].image.size)
    ```
    """

    def __init__(
        self,
        source: Union[Path, str, List[str], None] = None,
        glob_pattern: Union[str, List[str], None] = None,
    ) -> None:
        """Constructor for ImageFolder.

        Parameters
        ----------
        source
            A list of file paths or the path to the folder path that contains the images.
            A folder path can be either an absolute or relative folder path, in `str` or `Path` type. E.g. "/home/user/images".
            If you provide a folder path, all the files directly within the folder will be read (including non-image files).
            Nested files and sub-directories will be ignored.
            Consider using `glob_pattern` if you need to:
              1. filter out unwanted files, e.g. your folder has both image and non-image files
              2. read nested image files, e.g. `/home/user/images/**/*.jpg`.
            The ordering of images is based on the file name alphabetically if source is a folder path.
            If source is a list of files, the order of the input files is preserved.
            Currently only local file paths are supported.
        glob_pattern
            One or more python glob pattern(s) to grab only wanted files in the folder. E.g. "/home/user/images/*.jpg".
            NOTE: If `glob_pattern` is provided, the `source` parameter is ignored.
            For more information about glob pattern, see https://docs.python.org/3/library/glob.html

        """
        self._source = source
        self._image_paths: List[str] = []

        if source is None and glob_pattern is None:
            raise ValueError("Either 'source' or 'glob_pattern' must be provided.")
        if glob_pattern is not None:
            if isinstance(glob_pattern, str):
                glob_pattern = [glob_pattern]
            for pattern in glob_pattern:
                self._image_paths.extend(list(glob.glob(pattern, recursive=True)))
            self._image_paths.sort()
        elif isinstance(source, list):
            self._image_paths = source
        else:
            assert isinstance(source, str) or isinstance(source, Path)
            p = Path(source)
            if not p.exists():
                raise ValueError(f"Path '{p}' does not exist.")
            self._image_paths = [str(x) for x in p.glob("*") if x.is_file()]
            self._image_paths.sort()

    def __iter__(self) -> IteratorType[FrameSet]:
        for img_path in self._image_paths:
            meta = {"image_path": img_path}
            yield FrameSet.from_image(img_path, metadata=meta)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __repr__(self) -> str:
        return str(self._image_paths)

    @property
    def image_paths(self) -> List[str]:
        """Returns a list of image paths."""
        return self._image_paths
