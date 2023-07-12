from functools import lru_cache
import tempfile
from pathlib import Path
from typing import Callable, Union


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
        shell = get_ipython().__class__.__name__  # type: ignore
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


def read_from_notebook_webcam(webcam_source: Union[str, int] = 0) -> Callable[[], str]:
    """Return a function that reads an image from the webcam in notebook."""
    # Define function to acquire images either directly from the local webcam (i.e. jupyter notebook)or from the web browser (i.e. collab)
    local_cache_dir = Path(tempfile.mkdtemp())
    filename = str(local_cache_dir / "photo.jpg")
    # Detect if we are running on Google's colab
    try:
        from base64 import b64decode

        from google.colab.output import eval_js  # type: ignore
        from IPython.display import Javascript, display

        def take_photo() -> str:
            quality = 0.8
            js = Javascript(
                """
            async function takePhoto(quality) {
                const div = document.createElement('div');
                const capture = document.createElement('button');
                capture.textContent = 'Capture';
                div.appendChild(capture);

                const video = document.createElement('video');
                video.style.display = 'block';
                const stream = await navigator.mediaDevices.getUserMedia({video: true});

                document.body.appendChild(div);
                div.appendChild(video);
                video.srcObject = stream;
                await video.play();

                // Resize the output to fit the video element.
                google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

                // Wait for Capture to be clicked.
                await new Promise((resolve) => capture.onclick = resolve);

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                return canvas.toDataURL('image/jpeg', quality);
            }
            """
            )
            display(js)
            data = eval_js("takePhoto({})".format(quality))
            binary = b64decode(data.split(",")[1])
            with open(filename, "wb") as f:
                f.write(binary)
                return filename

    except ModuleNotFoundError:
        # Capture image from local webcam using OpenCV
        import cv2

        def take_photo() -> str:
            cam = cv2.VideoCapture(webcam_source)
            cv2.namedWindow("Press space to take photo")
            cv2.startWindowThread()
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    exit()
                cv2.imshow("Press space to take photo", frame)
                k = cv2.waitKey(1)
                if k % 256 == 32:
                    # SPACE pressed
                    cv2.imwrite(filename, frame)
                    break
            cam.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return filename

    return take_photo
