# audio-utils

A python library to handle wav's.

## Methods

- detect & remove silence
- detect & remove overamplification
- resampling
- concatenating
- normalizing
- stereo to mono conversion

Support for: INT32, INT32, FLOAT32 and FLOAT64 wavs

## Setup

Checkout this repository:

```sh
git clone https://github.com/stefantaubert/audio-utils.git
cd audio-utils
python3.8 -m pip install pipenv
python3.8 -m pipenv install --dev
```

### Add to another project

In the destination project run:

```sh
# if not already done:
pip install --user pipenv --python 3.8
# add reference
pipenv install -e git+https://github.com/stefantaubert/audio-utils.git@master#egg=audio_utils
```

## Dev

update setup.py with shell and `pipenv-setup sync`
see [details](https://pypi.org/project/pipenv-setup/)

## Notes

`Python 3.9` is not supported because `resampy` does not install on `Python 3.9` with dependency `llvmlite`. More [details](https://github.com/numba/numba/issues/6345).