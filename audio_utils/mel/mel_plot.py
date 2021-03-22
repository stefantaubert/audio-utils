from typing import List, Optional, Tuple

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from audio_utils.audio import get_sample_count
from matplotlib.figure import Figure


def concatenate_mels(audios: List[torch.Tensor], sentence_pause_s: float, sampling_rate: int) -> torch.Tensor:
  sentence_pause_samples_count = get_sample_count(sampling_rate, sentence_pause_s)
  return concatenate_mels_core(audios, sentence_pause_samples_count)


def concatenate_mels_core(audios: List[torch.Tensor], sentence_pause_samples_count: int = 0) -> torch.Tensor:
  if len(audios) == 1:
    return audios[0]

  dt = audios[0].dtype
  dev = audios[0].device
  size = audios[0].size()
  sentence_pause_samples = torch.zeros(
    [sentence_pause_samples_count, size[1], size[2]], dtype=dt, device=dev)
  output = torch.Tensor([], dtype=dt, device=dev)
  conc = []
  for audio in audios[:-1]:
    conc.append(audio)
    conc.append(sentence_pause_samples)
  conc.append(audios[-1])
  output = torch.cat(tuple(conc), dim=0)
  return output


def plot_melspec(mel: np.ndarray, mel_dim_x=16, mel_dim_y=5, factor=1, title=None):
  height, width = mel.shape
  width_factor = width / 1000
  _, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  axes.set_title(title)
  axes.set_yticks(np.arange(0, height, step=10))
  axes.set_xticks(np.arange(0, width, step=100))
  axes.set_xlabel("Samples")
  axes.set_ylabel("Freq. channel")
  axes.imshow(mel, aspect='auto', origin='lower', interpolation='none')
  return axes


def plot_melspec_np(mel: np.ndarray, mel_dim_x: int = 16, mel_dim_y: int = 5, factor: int = 1, title: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
  height, width = mel.shape
  width_factor = width / 1000
  fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(mel_dim_x * factor * width_factor, mel_dim_y * factor),
  )

  img = axes.imshow(
    X=mel,
    aspect='auto',
    origin='lower',
    interpolation='none'
  )

  axes.set_yticks(np.arange(0, height, step=5))
  axes.set_xticks(np.arange(0, width, step=50))
  axes.xaxis.set_major_locator(ticker.NullLocator())
  axes.yaxis.set_major_locator(ticker.NullLocator())
  plt.tight_layout()  # font logging occurs here
  figa_core = figure_to_numpy_rgb(fig)

  fig.colorbar(img, ax=axes)
  axes.xaxis.set_major_locator(ticker.AutoLocator())
  axes.yaxis.set_major_locator(ticker.AutoLocator())

  if title is not None:
    axes.set_title(title)
  axes.set_xlabel("Frames")
  axes.set_ylabel("Freq. channel")
  plt.tight_layout()  # font logging occurs here
  figa_labeled = figure_to_numpy_rgb(fig)
  plt.close()

  return figa_core, figa_labeled


def figure_to_numpy_rgb(figure: Figure) -> np.ndarray:
  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
  return data
