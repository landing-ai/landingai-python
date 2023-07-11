"""This module contains the telemetry configuration and APIs for the landingai package (intented for internal use only)."""


import os
import platform
import sys
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Dict


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
    if _is_running_in_colab():
        runtime = "colab"
    elif _is_running_in_notebook():
        runtime = "notebook"
    elif _is_running_in_streamlit():
        runtime = "streamlit"
    elif is_running_in_pytest():
        runtime = "pytest"
    else:
        runtime = Path(os.environ.get("_", "unknown")).name
    return runtime


def _is_running_in_colab() -> bool:
    """Return True if the code is running in a Google Colab notebook."""
    try:
        return get_ipython().__class__.__module__ == "google.colab._shell"  # type: ignore
    except NameError:
        return False  # Probably standard Python interpreter


def _is_running_in_notebook() -> bool:
    """Return True if the code is running in a Jupyter notebook."""
    try:
        # See: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def _is_running_in_streamlit() -> bool:
    """Return True if the code is running in a streamlit App."""
    # See: https://discuss.streamlit.io/t/how-to-check-if-code-is-run-inside-streamlit-and-not-e-g-ipython/23439/2
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False
