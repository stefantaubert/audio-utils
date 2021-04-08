from typing import Tuple

import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def align_mel_spectograms_with_dtw(mel_spec_1: np.ndarray, mel_spec_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
  mel_spec_1, mel_spec_2 = mel_spec_1.T, mel_spec_2.T
  dist, path = fastdtw(mel_spec_1, mel_spec_2, dist=euclidean)
  path_for_mel_spec_1 = list(map(lambda l: l[0], path))
  path_for_mel_spec_2 = list(map(lambda l: l[1], path))
  aligned_mel_spec_1 = mel_spec_1[path_for_mel_spec_1]
  aligned_mel_spec_2 = mel_spec_2[path_for_mel_spec_2]
  final_frame_number = len(path)
  msd = dist / final_frame_number
  assert final_frame_number == aligned_mel_spec_1.shape[1]
  assert final_frame_number == aligned_mel_spec_2.shape[1]
  return aligned_mel_spec_1, aligned_mel_spec_2, msd
