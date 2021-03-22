from setuptools import find_packages, setup

setup(
    name="audio_utils",
    version="1.0.0",
    url="https://github.com/stefantaubert/audio-utils.git",
    author="Stefan Taubert",
    author_email="stefan.taubert@posteo.de",
    description="Utils for audio processing",
    packages=["audio_utils"],
    install_requires=[
        "librosa",
        "matplotlib",
        "numpy",
        "resampy",
        "scikit-learn",
        "scipy",
        "torch<=1.7.1",
        "tqdm",
    ],
)
