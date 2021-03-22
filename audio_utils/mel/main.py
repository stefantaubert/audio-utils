import random
from typing import Tuple

import numpy as np
import torch
from audio_utils.audio import wav_to_float32


def mel_to_numpy(mel: torch.Tensor) -> np.ndarray:
  mel = mel.squeeze(0)
  mel = mel.cpu()
  mel_np: np.ndarray = mel.numpy()
  return mel_np


def wav_to_float32_tensor(path: str) -> Tuple[torch.Tensor, int]:
  wav, sampling_rate = wav_to_float32(path)
  wav_tensor = torch.FloatTensor(wav)

  return wav_tensor, sampling_rate


def get_wav_tensor_segment(wav_tensor: torch.Tensor, segment_length: int) -> torch.Tensor:
  if wav_tensor.size(0) >= segment_length:
    max_audio_start = wav_tensor.size(0) - segment_length
    audio_start = random.randint(0, max_audio_start)
    wav_tensor = wav_tensor[audio_start:audio_start + segment_length]
  else:
    fill_size = segment_length - wav_tensor.size(0)
    wav_tensor = torch.nn.functional.pad(wav_tensor, (0, fill_size), 'constant').data

  return wav_tensor
