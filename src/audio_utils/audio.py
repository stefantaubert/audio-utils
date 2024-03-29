import random
from math import ceil, inf, log10
from typing import List, Tuple

import numpy as np
from resampy import resample as resamply_resample
from scipy.io.wavfile import read, write

# import torch

FLOAT32_64_MIN_WAV = -1.0
FLOAT32_64_MAX_WAV = 1.0
INT16_MIN = np.iinfo(np.int16).min  # -32768 = -(2**15)
INT16_MAX = np.iinfo(np.int16).max  # 32767 = 2**15 - 1
INT32_MIN = np.iinfo(np.int32).min  # -2147483648 = -(2**31)
INT32_MAX = np.iinfo(np.int32).max  # 2147483647 = 2**31 - 1


def get_dBFS(wav, max_value) -> float:
  value = np.sqrt(np.mean((wav / max_value)**2))
  if value == 0:
    return -inf
  result = 20 * log10(value)
  return result


def detect_leading_silence(wav: np.array, silence_threshold: float, chunk_size: int, buffer: int):
  assert chunk_size > 0
  if chunk_size > len(wav):
    chunk_size = len(wav)

  trim = 0
  max_value = -1 * get_min_value(wav.dtype)
  while get_dBFS(wav[trim:trim + chunk_size], max_value) < silence_threshold and trim < len(wav):
    trim += chunk_size

  if trim >= buffer:
    trim -= buffer
  else:
    trim = 0

  return trim


def remove_silence(
    wav,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start: int,
    buffer_end: int
  ):

  start_trim = detect_leading_silence(
    wav=wav,
    silence_threshold=threshold_start,
    chunk_size=chunk_size,
    buffer=buffer_start
  )

  wav_reversed = wav[::-1]
  end_trim = detect_leading_silence(
    wav=wav_reversed,
    silence_threshold=threshold_end,
    chunk_size=chunk_size,
    buffer=buffer_end
  )

  wav = wav[start_trim:len(wav) - end_trim]
  return wav


def get_closest_sample_rate_s(seconds: float, sample_rate: int, precision: int = 4) -> float:
  samples = s_to_samples(seconds, sample_rate, precision)
  result = samples_to_s(samples, sample_rate)
  return result

def s_to_samples(s: float, sampling_rate: int, precision: int = 4) -> int:
  res = ceil(round(s * sampling_rate, precision))
  return res


def ms_to_samples(ms, sampling_rate, precision: int = 4) -> int:
  return s_to_samples(ms / 1000, sampling_rate, precision)


def samples_to_ms(samples, sampling_rate) -> float:
  return samples_to_s(samples, sampling_rate) * 1000


def samples_to_s(samples: int, sampling_rate: int) -> float:
  res = samples / sampling_rate
  return res


def remove_silence_file(
    in_path: str,
    out_path: str,
    chunk_size: int,
    threshold_start: float,
    threshold_end: float,
    buffer_start_ms: int,
    buffer_end_ms: int
  ):

  sampling_rate, wav = read(in_path)

  buffer_start = ms_to_samples(buffer_start_ms, sampling_rate)
  buffer_end = ms_to_samples(buffer_end_ms, sampling_rate)

  wav = remove_silence(
    wav=wav,
    chunk_size=chunk_size,
    threshold_start=threshold_start,
    threshold_end=threshold_end,
    buffer_start=buffer_start,
    buffer_end=buffer_end
  )
  new_duration = get_duration_s(wav, sampling_rate)

  write(filename=out_path, rate=sampling_rate, data=wav)

  return new_duration


def float_to_wav(wav, path, dtype=np.int16, sample_rate=22050):
  # denoiser_out is float64
  # waveglow_out is float32

  wav = convert_wav(wav, dtype)

  write(filename=path, rate=sample_rate, data=wav)


def convert_wav(wav, to_dtype):
  '''
  if the wav is overamplified the result will also be overamplified.
  '''
  if wav.dtype != to_dtype:
    wav = wav / (-1 * get_min_value(wav.dtype)) * get_max_value(to_dtype)
    if to_dtype in (np.int16, np.int32):
      # the default seems to be np.fix instead of np.round on wav.astype()
      wav = np.round(wav, 0)
    wav = wav.astype(to_dtype)

  return wav


def fix_overamplification(wav):
  is_overamplified = is_overamp(wav)
  if is_overamplified:
    wav = normalize_wav(wav)
  return wav


def get_max_value(dtype):
  # see wavfile.write() max positive eg. on 16-bit PCM is 32767
  if dtype == np.int16:
    return INT16_MAX

  if dtype == np.int32:
    return INT32_MAX

  if dtype in (np.float32, np.float64):
    return FLOAT32_64_MAX_WAV

  assert False


def get_min_value(dtype):
  if dtype == np.int16:
    return INT16_MIN

  if dtype == np.int32:
    return INT32_MIN

  if dtype == np.float32 or dtype == np.float64:
    return FLOAT32_64_MIN_WAV

  assert False


def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))


def concatenate_audios(audios: List[np.ndarray], sentence_pause_s: float, sampling_rate: int) -> np.array:
  sentence_pause_samples_count = get_sample_count(sampling_rate, sentence_pause_s)
  return concatenate_audios_core(audios, sentence_pause_samples_count)


def concatenate_audios_core(audios: List[np.ndarray], sentence_pause_samples_count: int = 0) -> np.ndarray:
  """Concatenates the np.ndarray list on the last axis."""
  if len(audios) == 1:
    cpy = np.array(audios[0])
    return cpy

  pause_shape = list(audios[0].shape)
  pause_shape[-1] = sentence_pause_samples_count
  sentence_pause_samples = np.zeros(tuple(pause_shape))
  conc = []
  for audio in audios[:-1]:
    conc.append(audio)
    conc.append(sentence_pause_samples)
  conc.append(audios[-1])
  output = np.concatenate(tuple(conc), axis=-1)
  return output


def normalize_file(in_path, out_path):
  sampling_rate, wav = read(in_path)
  wav = normalize_wav(wav)
  write(filename=out_path, rate=sampling_rate, data=wav)


def normalize_wav(wav: np.ndarray):
  # Mono or stereo is supported.

  if wav.dtype == np.int16 and np.min(wav) == get_min_value(np.int16):
    return wav
  if wav.dtype == np.int32 and np.min(wav) == get_min_value(np.int32):
    return wav

  wav_abs = np.abs(wav)
  max_val = np.max(wav_abs)
  is_div_by_zero = max_val == 0
  max_possible_value = get_max_value(wav.dtype)
  is_already_normalized = max_val == max_possible_value
  # on int16 resulting min wav value would be max. -32767 not -32768 (which would be possible with wavfile.write) maybe later TODO

  if not is_already_normalized and not is_div_by_zero:
    orig_dtype = wav.dtype
    wav_float = wav.astype(np.float32)
    wav_float = wav_float * max_possible_value / max_val
    if orig_dtype == np.int16 or orig_dtype == np.int32:
      # the default seems to be np.fix instead of np.round on wav.astype()
      # 32766.998 gets 32767 because of float unaccuracy
      wav_float = np.round(wav_float, 0)
    wav = wav_float.astype(orig_dtype)

  assert np.max(np.abs(wav)) == max_possible_value or np.max(np.abs(wav)) == 0
  assert not is_overamp(wav)

  return wav


def wav_to_float32(path: str) -> Tuple[np.float, int]:
  sampling_rate, wav = read(path)
  wav = convert_wav(wav, np.float32)
  return wav, sampling_rate


# TODO: does not really work that check
def is_overamp(wav: np.ndarray) -> bool:
  lowest_value = get_min_value(wav.dtype)
  highest_value = get_max_value(wav.dtype)
  wav_min = np.min(wav)
  wav_max = np.max(wav)
  is_overamplified = wav_min < lowest_value or wav_max > highest_value
  return is_overamplified


def resample_core(wav: np.ndarray, sr: int, new_rate: int) -> np.ndarray:
  if not check_wav_is_mono(wav):
    raise Exception("It is only possible to resample mono-channel wavs.")

  if sr != new_rate:
    origin_dtype = wav.dtype
    wav_float = convert_wav(wav, np.float32)
    wav_float = resamply_resample(wav_float, sr, new_rate)
    # if a.min was -1 before resample it would be smaller than -1 (bug in resample)
    wav_float = fix_overamplification(wav_float)
    wav = convert_wav(wav_float, origin_dtype)
  return wav


def upsample_file(origin: str, dest: str, new_rate: int) -> None:
  sampling_rate, wav = read(origin)
  wav = resample_core(wav, sampling_rate, new_rate)
  write(filename=dest, rate=new_rate, data=wav)


def check_wav_is_stereo(wav: np.ndarray) -> bool:
  channels_axis = 1
  wav_is_stereo = len(wav.shape) == 2 and wav.shape[channels_axis] == 2
  return wav_is_stereo


def check_wav_is_mono(wav: np.ndarray) -> bool:
  wav_is_mono = len(wav.shape) == 1
  return wav_is_mono


def stereo_to_mono(wav: np.ndarray) -> np.ndarray:
  assert check_wav_is_stereo(wav)

  origin_dtype = wav.dtype
  wav_float = convert_wav(wav, np.float32)
  channels_axis = 1
  wav_float = wav_float.sum(axis=channels_axis) / 2
  wav = convert_wav(wav_float, origin_dtype)
  return wav


def stereo_to_mono_file(origin: str, dest: str) -> None:
  sampling_rate, wav = read(origin)
  wav = stereo_to_mono(wav)
  write(filename=dest, rate=sampling_rate, data=wav)


def get_duration_s(wav, sampling_rate) -> float:
  return get_duration_s_samples(len(wav), sampling_rate)


def get_duration_s_samples(samples: int, sampling_rate: int) -> float:
  duration = samples / sampling_rate
  return duration


def get_duration_s_file(wav_path) -> float:
  sampling_rate, wav = read(wav_path)
  return get_duration_s(wav, sampling_rate)


# def mel_to_numpy(mel: torch.Tensor) -> np.ndarray:
#   mel = mel.squeeze(0)
#   mel = mel.cpu()
#   mel_np: np.ndarray = mel.numpy()
#   return mel_np


# def wav_to_float32_tensor(path: str) -> Tuple[torch.Tensor, int]:
#   wav, sampling_rate = wav_to_float32(path)
#   wav_tensor = torch.FloatTensor(wav)

#   return wav_tensor, sampling_rate


# def get_wav_tensor_segment(wav_tensor: torch.Tensor, segment_length: int) -> torch.Tensor:
#   if wav_tensor.size(0) >= segment_length:
#     max_audio_start = wav_tensor.size(0) - segment_length
#     audio_start = random.randint(0, max_audio_start)
#     wav_tensor = wav_tensor[audio_start:audio_start + segment_length]
#   else:
#     fill_size = segment_length - wav_tensor.size(0)
#     wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data

#   return wav_tensor
