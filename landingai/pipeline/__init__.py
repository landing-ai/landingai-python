"""The vision pipeline abstraction helps chain image processing operations as
sequence of steps. Each step consumes and produces a `FrameSet` which typically
contains a source image and derivative metadata and images."""

# Import image_source to enable IDE auto completion
import landingai.pipeline.image_source as image_source  # noqa: F401
import landingai.pipeline.postprocessing as postprocessing  # noqa: F401

from .frameset import FrameSet  # noqa: F401
