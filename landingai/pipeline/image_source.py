"""A module that provides a set of abstractions and APIs for reading images from different sources."""

import glob
from collections.abc import Iterator
from pathlib import Path
from typing import Iterator as IteratorType
from typing import List, Optional, Union

from landingai.vision_pipeline import FrameSet

_SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "tiff", "bmp", "gif"]


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
    folder = ImageFolder(glob_pattern="/home/user/images/*.jpg")
    for image_batch in folder:
        print(image_batch[0].image.size)
    ```
    """

    def __init__(
        self,
        source: Union[Path, str, List[str], None] = None,
        glob_pattern: Optional[str] = None,
    ) -> None:
        """Constructor for ImageFolder.

        Parameters
        ----------
        source
            A list of file names or the path to the folder path that contains the images.
            A folder path can be either a absolute or relative folder path, in `str` or `Path` type. E.g. "/home/user/images"
            Currently only supports local file paths.
            Within the folder, the images can be in any of the supported formats: jpg, jpeg, png, tiff, bmp, gif. Other file formats will be ignored.
            The ordering of images is based on the file name alphabetically if source is a folder path.
            If source is a list of files, the order of the input files is preserved.
        glob_pattern
            A python glob pattern (case sensitive) to grab only wanted files in the folder. E.g. "/home/user/images/*.jpg"
            NOTE: If `glob_pattern` is provided, the `source` parameter is ignored.
            For more information about glob pattern, see https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob

        """
        self._source = source
        self._image_paths: List[str] = []
        self._index = 0
        if source is None and glob_pattern is None:
            raise ValueError("Either 'source' or 'glob_pattern' must be provided.")
        if glob_pattern is not None:
            self._image_paths = list(glob.glob(glob_pattern))
            sorted(self._image_paths)
        elif isinstance(source, list):
            self._image_paths = source
        else:
            assert isinstance(source, str) or isinstance(source, Path)
            folder_path = Path(source)
            for ext in _SUPPORTED_IMAGE_FORMATS:
                self._image_paths.extend([str(p) for p in folder_path.glob(f"*.{ext}")])
                self._image_paths.extend(
                    [str(p) for p in folder_path.glob(f"*.{ext.upper()}")]
                )
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
