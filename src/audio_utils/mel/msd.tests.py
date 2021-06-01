import unittest

import numpy as np
import torch
from audio_utils.mel.msd import align_mels_with_dtw


class UnitTests(unittest.TestCase):

  def test_align_mels_with_dtw(self):
    np.random.seed(1)
    a = np.random.rand(80, 10)
    b = np.random.rand(80, 70)
    x, y, z, path_a, path_b = align_mels_with_dtw(a, b)

    self.assertEqual(x.shape, y.shape)
    self.assertEqual(246.00328760393276, z)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
