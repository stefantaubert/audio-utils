from typing import List, Tuple

import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def align_mels_with_dtw(mel_spec_1: np.ndarray, mel_spec_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int]]:
  mel_spec_1, mel_spec_2 = mel_spec_1.T, mel_spec_2.T
  dist, path = fastdtw(mel_spec_1, mel_spec_2, dist=euclidean)
  path_for_mel_spec_1 = list(map(lambda l: l[0], path))
  path_for_mel_spec_2 = list(map(lambda l: l[1], path))
  aligned_mel_spec_1 = mel_spec_1[path_for_mel_spec_1].T
  aligned_mel_spec_2 = mel_spec_2[path_for_mel_spec_2].T
  return aligned_mel_spec_1, aligned_mel_spec_2, dist, path_for_mel_spec_1, path_for_mel_spec_2


def get_msd(dist: float, total_frame_number: int) -> float:
  msd = dist / total_frame_number
  return msd
