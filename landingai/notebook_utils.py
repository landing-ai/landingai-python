"""This module contains common notebook utilities that are used across the example notebooks in this repo.
It's only intended for examples provided by this repo. When using the SDK in your own project, you don't need to use this module.
"""

from functools import lru_cache


def is_running_in_colab_notebook() -> bool:
    """Return True if the code is running in a Google Colab notebook."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__module__ == "google.colab._shell"  # type: ignore
    except ImportError:
        return False  # Probably standard Python interpreter


def is_running_in_jupyter_notebook() -> bool:
    """Return True if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        # See: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except ImportError:
        return False  # Probably standard Python interpreter


@lru_cache(maxsize=None)
def is_running_in_notebook() -> bool:
    """Return True if the code is running in a notebook."""
    return is_running_in_colab_notebook() or is_running_in_jupyter_notebook()


def display_video(path_to_file: str):  # type: ignore
    """Return a notebook-independent video object that can be shown on Jupyter and Colab."""
    if is_running_in_colab_notebook():
        from IPython.display import HTML
        from base64 import b64encode

        mp4 = open(path_to_file, "rb").read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML(
            """
        <video width=400 controls>
            <source src="%s" type="video/mp4">
        </video>
        """
            % data_url
        )
    else:
        from IPython.display import Video

        return Video(path_to_file)
