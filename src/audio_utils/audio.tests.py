import math
import os
import tempfile
import unittest

import numpy as np
from scipy.io.wavfile import write

from audio_utils.audio import *


class UnitTests(unittest.TestCase):

  def test_get_dBFS_min_int_is_zero(self):
    res = get_dBFS(np.array([-32768]), 32768)

    self.assertEqual(0.0, res)

  def test_get_dBFS_max_int_is_almost_zero(self):
    res = get_dBFS(np.array([32767]), 32768)

    self.assertTrue(res < -0.0001)
    self.assertTrue(res > -0.0005)

  def test_get_dBFS_int_is_inf(self):
    res = get_dBFS(np.array([0.0]), 32768)

    self.assertEqual(-math.inf, res)

  def test_get_dBFS_max_float_is_zero(self):
    res = get_dBFS(np.array([1.0]), 1.0)

    self.assertEqual(0.0, res)

  def test_get_dBFS_min_float_is_zero(self):
    res = get_dBFS(np.array([-1.0]), 1.0)

    self.assertEqual(0.0, res)

  def test_get_dBFS_float_is_inf(self):
    res = get_dBFS(np.array([0.0]), 1.0)

    self.assertEqual(-math.inf, res)

  def test_detect_silence_float32_chunksize_bigger_wavlength_silence(self):
    samples = 50
    audio = np.array([0] * samples, dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0, chunk_size=100, buffer=0)

    self.assertEqual(samples, result)

  def test_detect_silence_float32_chunksize_bigger_wavlength_data(self):
    samples = 50
    audio = np.array([0.5] * samples, dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0, chunk_size=100, buffer=0)

    self.assertEqual(0, result)

  def test_detect_silence_float32_no_buffer(self):
    pause_samples = 50
    audio = np.array([0] * pause_samples + [0.5], dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0, chunk_size=1, buffer=0)

    self.assertEqual(pause_samples, result)

  def test_detect_silence_float32_buffer_smaller_trim(self):
    pause_samples = 50
    audio = np.array([0] * pause_samples + [0.5], dtype=np.float32)

    buffer_size = 10
    result = detect_leading_silence(audio, silence_threshold=-50.0,
                                    chunk_size=1, buffer=buffer_size)

    self.assertEqual(pause_samples - buffer_size, result)

  def test_detect_silence_float32_buffer_equals_trim(self):
    pause_samples = 50
    audio = np.array([0] * pause_samples + [0.5], dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0,
                                    chunk_size=1, buffer=pause_samples)

    self.assertEqual(0, result)

  def test_detect_silence_float32_buffer_greater_trim(self):
    pause_samples = 50
    audio = np.array([0] * pause_samples + [0.5], dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0,
                                    chunk_size=1, buffer=pause_samples + 1)

    self.assertEqual(0, result)

  def test_detect_silence_float32_silence_track_is_trimmed(self):
    pause_samples = 500
    audio = np.array([0] * pause_samples, dtype=np.float32)

    result = detect_leading_silence(audio, silence_threshold=-50.0, chunk_size=1, buffer=0)

    self.assertEqual(pause_samples, result)

  def test_remove_start_and_end_silence_float32(self):
    pause_samples = 50
    audio = np.array([0] * pause_samples + [0.5] + [0] * pause_samples, dtype=np.float32)

    result = remove_silence(audio, threshold_start=-50.0, threshold_end=-
                            50.0, chunk_size=1, buffer_start=0, buffer_end=0)

    self.assertEqual(1, len(result))

  def test_concatenate_two_dim(self):
    a = np.zeros((80, 34))
    b = np.zeros((80, 66))
    x = concatenate_audios_core([a, b])

    self.assertEqual((80, 100), x.shape)

  def test_concatenate_no_pause(self):
    a = np.zeros(71)
    b = np.zeros(29)
    x = concatenate_audios_core([a, b])

    self.assertEqual(100, len(x))

  def test_concatenate_one_pause(self):
    a = np.zeros(71)
    b = np.zeros(29)
    x = concatenate_audios_core([a, b], sentence_pause_samples_count=50)

    self.assertEqual(100 + 50, len(x))

  def test_concatenate_two_pause(self):
    a = np.zeros(71)
    b = np.zeros(29)
    c = np.zeros(30)
    x = concatenate_audios_core([a, b, c], sentence_pause_samples_count=50)

    self.assertEqual(130 + 100, len(x))

  def test_concatenate_one_element(self):
    a = np.zeros(100)
    x = concatenate_audios_core([a], sentence_pause_samples_count=50)

    self.assertEqual(100, len(x))

  def test_ms_to_samples_500ms_22050sr(self):
    sr = 22050
    duration_ms = 500

    result = ms_to_samples(duration_ms, sr)

    self.assertEqual(sr / 2, result)

  def test_ms_to_samples_1s_22050sr(self):
    sr = 22050
    duration_ms = 1000

    result = ms_to_samples(duration_ms, sr)

    self.assertEqual(sr, result)

  def test_ms_to_samples_0s_22050sr(self):
    sr = 22050
    duration_ms = 0

    result = ms_to_samples(duration_ms, sr)

    self.assertEqual(0, result)

  def test_ms_to_samples_1s_0sr(self):
    sr = 0
    duration_ms = 1000

    result = ms_to_samples(duration_ms, sr)

    self.assertEqual(0, result)

  def test_get_duration_float32(self):
    sr = 16000
    duration_ms = 500
    num_samples = int(duration_ms * sr / 1000.0)
    audio = np.array([0.0] * num_samples, dtype=np.float32)
    tmp = tempfile.mktemp()
    write(tmp, sr, audio)

    duration_s = get_duration_s_file(tmp)

    os.remove(tmp)

    self.assertEqual(0.5, duration_s)

  def test_resample_float32_16000_to_22050(self):
    old_sr = 16000
    new_sr = 22050
    duration_ms = 500
    old_num_samples = int(duration_ms * old_sr / 1000.0)
    new_num_samples = int(duration_ms * new_sr / 1000.0)
    audio = np.array([0.0] * old_num_samples, dtype=np.float32)

    new_audio = resample_core(audio, old_sr, new_sr)

    self.assertEqual(new_num_samples, len(new_audio))
    self.assertEqual(0, np.min(new_audio))
    self.assertEqual(0, np.max(new_audio))

  def test_resample_float32_22050_to_16000(self):
    old_sr = 22050
    new_sr = 16000
    duration_ms = 500
    old_num_samples = int(duration_ms * old_sr / 1000.0)
    new_num_samples = int(duration_ms * new_sr / 1000.0)
    audio = np.array([0.0] * old_num_samples, dtype=np.float32)

    new_audio = resample_core(audio, old_sr, new_sr)

    self.assertEqual(new_num_samples, len(new_audio))
    self.assertEqual(0, np.min(new_audio))
    self.assertEqual(0, np.max(new_audio))

  def test_resample_float32_16000_to_22050_with_overamp(self):
    old_sr = 16000
    new_sr = 22050
    duration_ms = 500
    old_num_samples = int(duration_ms * old_sr / 1000.0)
    new_num_samples = int(duration_ms * new_sr / 1000.0)
    audio = np.array(
      [1.0] * old_num_samples +
      [0] * old_num_samples +
      [-1.0] * old_num_samples, dtype=np.float32)

    new_audio = resample_core(audio, old_sr, new_sr)

    self.assertEqual(3 * new_num_samples, len(new_audio))
    self.assertEqual(-1.0, np.min(new_audio))
    self.assertEqual(1.0, np.max(new_audio))

  def test_normalize_float32(self):
    audio = np.array([-0.5, 0, 0.5], dtype=np.float32)

    new_audio = normalize_wav(audio)

    self.assertEqual(-1.0, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(1.0, new_audio[2])

  def test_normalize_overamp_float32(self):
    audio = np.array([-1.5, 0, 1.5], dtype=np.float32)

    new_audio = normalize_wav(audio)

    self.assertEqual(-1.0, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(1.0, new_audio[2])

  def test_normalize_int16(self):
    audio = np.array([-100, 0, 100], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(-32767, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(32767, new_audio[2])

  def test_normalize_int16_min_value(self):
    audio = np.array([-32768], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(-32768, new_audio[0])

  def test_normalize_int16_max_value(self):
    audio = np.array([32767], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(32767, new_audio[0])

  def test_normalize_int32_min_value(self):
    audio = np.array([-2147483648], dtype=np.int32)

    new_audio = normalize_wav(audio)

    self.assertEqual(-2147483648, new_audio[0])

  def test_normalize_int32_min_int16_value_is_normalized(self):
    audio = np.array([-32768], dtype=np.int32)

    new_audio = normalize_wav(audio)

    self.assertEqual(-2147483647, new_audio[0])

  def test_normalize_int32_max_value(self):
    audio = np.array([2147483647], dtype=np.int32)

    new_audio = normalize_wav(audio)

    self.assertEqual(2147483647, new_audio[0])

  def test_normalize_int16_rounding(self):
    audio = np.array([-18413], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(-32767, new_audio[0])

  def test_convert_float_int16_positive_is_rounded_up(self):
    audio = np.array([32766.99 / 32767], dtype=np.float32)

    new_audio = convert_wav(audio, np.int16)

    self.assertEqual(32767, new_audio[0])

  def test_convert_float_int16_negative_is_rounded_down(self):
    audio = np.array([-32766.99 / 32767], dtype=np.float32)

    new_audio = convert_wav(audio, np.int16)

    self.assertEqual(-32767, new_audio[0])

  def test_normalize_int16_already_normalized(self):
    audio = np.array([-32767, 0, 32767], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(-32767, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(32767, new_audio[2])

  def test_normalize_overamp_int16(self):
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    # np.abs(audio) = [-32768, 0, 32767] !!

    new_audio = normalize_wav(audio)

    self.assertEqual(-32768, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(32767, new_audio[2])

  def test_normalize_pause_int16(self):
    audio = np.array([0, 0, 0], dtype=np.int16)

    new_audio = normalize_wav(audio)

    self.assertEqual(0, new_audio[0])
    self.assertEqual(0, new_audio[1])
    self.assertEqual(0, new_audio[2])

  def test_is_overamp_max_val_float32_returns_false(self):
    audio = np.array([-1.0, 0, 1.0], dtype=np.float32)

    res = is_overamp(audio)

    self.assertFalse(res)

  def test_is_overamp_float32_tobigmin_returns_true(self):
    audio = np.array([-1.5, 0, 1.0], dtype=np.float32)

    res = is_overamp(audio)

    self.assertTrue(res)

  def test_is_overamp_float32_tobigmax_returns_true(self):
    audio = np.array([-1.0, 0, 1.5], dtype=np.float32)

    res = is_overamp(audio)

    self.assertTrue(res)

  def test_is_overamp_pause_int16(self):
    audio = np.array([0, 0, 0], dtype=np.int16)

    res = is_overamp(audio)

    self.assertFalse(res)

  def test_is_overamp_max_val_int16(self):
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    res = is_overamp(audio)

    self.assertFalse(res)

  def test_is_overamp_int16_true_is_ignored(self):
    audio = np.array([-32769, 0, 32768], dtype=np.int16)

    res = is_overamp(audio)

    # np min is -32768 and max is 32767 !!
    self.assertFalse(res)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
