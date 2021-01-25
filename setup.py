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
        "llvmlite; python_version >= '3.6'",
        "numba; python_version >= '3.6' and python_version < '3.9'",
        "numpy",
        "resampy",
        "scipy",
        "six; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "tqdm",
    ],
)
