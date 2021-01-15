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
        "llvmlite==0.35.0; python_version >= '3.6'",
        "numba==0.52.0; python_version >= '3.6' and python_version < '3.9'",
        "numpy==1.19.5",
        "resampy==0.2.2",
        "scipy==1.6.0",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "tqdm==4.56.0",
    ],
)
