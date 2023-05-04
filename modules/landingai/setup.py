from setuptools import setup

setup(
    name="landingai",
    version="0.0.1",
    description="Helper library for interacting with Landing AI LandingLens",
    author="Landing AI",
    author_email="dev@landing.ai",
    python_requires='>=3.8',
    install_requires=['opencv-python', 'numpy', 'requests'],
)
