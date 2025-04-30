# Edge Models

## Installation

- Python >= 3.10

Linux:
```
pip install -r requirements.txt
```

Windows:
```
pip install -r requirements_win.txt
```

Note: tflite-runtime is not available for Windows: https://github.com/google-ai-edge/LiteRT?tab=readme-ov-file#pypi-installation-requirements
so on Windows we suggest to use Tensorflow.

## Run

Basic:
```
python run_bundle_cls.py published_model_bundle.zip image.png
```

Enabling hardware acceleration (this is just an example, add your own library path):
```
python run_bundle_cls.py published_model_bundle.zip image.png /usr/lib/libvx_delegate.so
```
