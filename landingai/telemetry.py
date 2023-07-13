"""This module contains the telemetry configuration and APIs for the landingai package (intented for internal use only)."""


import os
import platform
import sys
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Dict

from landingai.notebook_utils import (
    is_running_in_colab_notebook,
    is_running_in_jupyter_notebook,
)
from landingai.st_utils import is_running_in_streamlit


@lru_cache(maxsize=None)
def get_runtime_environment_info() -> Dict[str, str]:
    """Return a set of runtime environment information in key value pairs."""
    return {
        "lib_type": "pylib",
        "lib_version": version("landingai"),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "runtime": _resolve_python_runtime(),
    }


@lru_cache(maxsize=None)
def is_running_in_pytest() -> bool:
    """Return True if the code is running in a pytest session."""
    # See: https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    return "pytest" in sys.modules


def _resolve_python_runtime() -> str:
    if is_running_in_colab_notebook():
        runtime = "colab"
    elif is_running_in_jupyter_notebook():
        runtime = "notebook"
    elif is_running_in_streamlit():
        runtime = "streamlit"
    elif is_running_in_pytest():
        runtime = "pytest"
    else:
        runtime = Path(os.environ.get("_", "unknown")).name
    return runtime
