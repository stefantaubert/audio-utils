from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from audio_utils.audio import get_dBFS, get_min_value


@dataclass
class Chunk:
  size: int
  is_silence: bool


def get_chunks(wav: np.ndarray, silence_boundary: float, chunk_size: int, min_silence_duration: int, min_content_duration: int, content_buffer_start: int, content_buffer_end: int) -> List[Chunk]:
  chunks = mask_silence(wav, silence_boundary, chunk_size)
  chunks = merge_same_coherent_chunks(chunks)
  chunks = merge_silent_chunks(chunks, min_silence_duration)
  chunks = merge_content_chunks(chunks, min_content_duration)
  chunks = add_content_start_buffer(chunks, content_buffer_start)
  chunks = add_content_end_buffer(chunks, content_buffer_end)
  return chunks


def add_content_start_buffer(chunks: List[Chunk], duration: int) -> List[Chunk]:
  content_mask = get_content_mask(chunks)
  tmp = add_start_buffer(chunks, content_mask, duration)
  tmp = remove_zero_duration_chunks(tmp)
  tmp = merge_same_coherent_chunks(tmp)
  return tmp


def add_start_buffer(chunks: List[Chunk], mask: List[bool], duration: int) -> List[Chunk]:
  assert len(chunks) == len(mask)
  if len(chunks) <= 1:
    return chunks

  for i in range(1, len(chunks)):
    current_mask = mask[i]
    current_chunk = chunks[i]
    previous_chunk = chunks[i - 1]
    if current_mask:
      reduce_duration = previous_chunk.size - max(0, previous_chunk.size - duration)
      # if it would be smaller than 0 there is a content therefore only the pause is removed
      previous_chunk.size -= reduce_duration
      current_chunk.size += reduce_duration

  return chunks


def add_content_end_buffer(chunks: List[Chunk], duration: int) -> List[Chunk]:
  content_mask = get_content_mask(chunks)
  tmp = add_end_buffer(chunks, content_mask, duration)
  tmp = remove_zero_duration_chunks(tmp)
  tmp = merge_same_coherent_chunks(tmp)
  return tmp


def add_end_buffer(chunks: List[Chunk], mask: List[bool], duration: int):
  assert len(chunks) == len(mask)
  if len(chunks) <= 1:
    return chunks

  for i in range(0, len(chunks) - 1):
    current_mask = mask[i]
    current_chunk = chunks[i]
    next_chunk = chunks[i + 1]
    if current_mask:
      reduce_duration = next_chunk.size - max(0, next_chunk.size - duration)
      # if it would be smaller than 0 there is a content therefore only the pause is removed
      next_chunk.size -= reduce_duration
      current_chunk.size += reduce_duration

  return chunks


def get_silence_mask(chunks: List[Chunk]) -> List[bool]:
  return [x.is_silence for x in chunks]


def get_content_mask(chunks: List[Chunk]) -> List[bool]:
  return [not x.is_silence for x in chunks]


def remove_zero_duration_chunks(chunks: List[Chunk]) -> List[Chunk]:
  res = [x for x in chunks if x.size > 0]
  return res


def merge_content_chunks(chunks: List[Chunk], min_duration: int) -> List[Chunk]:
  duration_mask = mask_too_short_entries(chunks, min_duration)
  process_mask = get_content_mask(chunks)
  mask = bool_and(duration_mask, process_mask)
  tmp = merge_marked_chunks(chunks, mask)
  tmp = merge_same_coherent_chunks(tmp)
  return tmp


def merge_silent_chunks(chunks: List[Chunk], min_duration: int) -> List[Chunk]:
  duration_mask = mask_too_short_entries(chunks, min_duration)
  process_mask = get_silence_mask(chunks)
  mask = bool_and(duration_mask, process_mask)
  tmp = merge_marked_chunks(chunks, mask)
  tmp = merge_same_coherent_chunks(tmp)
  return tmp


def mask_too_short_entries(chunks: List[Chunk], min_duration: int) -> List[bool]:
  res = [x.size < min_duration for x in chunks]
  return res


def bool_and(match_mask: List[bool], process_mask: List[bool]) -> List[bool]:
  res = [x and y for x, y in zip(match_mask, process_mask)]
  return res


def merge_marked_chunks(chunks: List[Chunk], merge_mask: List[bool]) -> List[Chunk]:
  assert len(chunks) == len(merge_mask)
  if len(chunks) == 1:
    return [chunks[0]]

  res: List[Chunk] = []
  merge_size = 0
  for i, _ in enumerate(chunks):
    current_chunk = chunks[i]
    merge_current_chunk = merge_mask[i]
    is_last_chunk = i == len(chunks) - 1
    if merge_current_chunk:
      if is_last_chunk:
        assert i - 1 >= 0
        assert merge_size == 0
        prev_chunk = chunks[i - 1]
        prev_chunk.size += current_chunk.size
      else:
        merge_size += current_chunk.size
    else:
      current_chunk.size += merge_size
      res.append(current_chunk)
      merge_size = 0

  return res


def merge_content_chunks_old(chunks: List[Chunk], min_duration: int) -> List[Chunk]:
  ''' Merges chunks with previous chunk if duration is smaller than min_duration. '''
  if len(chunks) <= 1:
    return chunks

  final_chunks: List[Chunk] = [chunks[0]]
  current_chunk: Chunk = None
  for i in range(1, len(chunks)):
    current_chunk = chunks[i]
    if current_chunk.is_silence:
      previous_chunk = chunks[i - 1]
      #assert not previous_chunk.is_silence
      merge_previous_chunk = previous_chunk.size < min_duration
      if merge_previous_chunk:
        current_chunk.size += previous_chunk.size
        final_chunks.append(current_chunk)
      else:
        final_chunks.append(previous_chunk)
        final_chunks.append(current_chunk)

  assert current_chunk is not None
  last_chunk = current_chunk

  if not last_chunk.is_silence:
    assert len(final_chunks) >= 1
    merge_last_chunk = last_chunk.size < min_duration
    if merge_last_chunk:
      previous_chunk = final_chunks[-1]
      previous_chunk.size += last_chunk.size
    else:
      final_chunks.append(last_chunk)

  return final_chunks


def merge_chunks(chunks: List[Chunk], min_duration: int, merge_silence: bool) -> List[Chunk]:
  ''' Merges chunks with previous chunk if duration is smaller than min_duration. '''
  if len(chunks) <= 1:
    return chunks

  final_chunks: List[Chunk] = list()
  current_chunk: Chunk = None
  for i in range(1, len(chunks)):
    current_chunk = chunks[i]
    process_chunk = (not merge_silence and current_chunk.is_silence) or (
      merge_silence and not current_chunk.is_silence)
    if process_chunk:
      previous_chunk = chunks[i - 1]
      #assert not previous_chunk.is_silence
      merge_previous_chunk = previous_chunk.size < min_duration
      if merge_previous_chunk:
        current_chunk.size += previous_chunk.size
        final_chunks.append(current_chunk)
      else:
        final_chunks.append(previous_chunk)
        final_chunks.append(current_chunk)

  if len(final_chunks) == 0:
    final_chunks.append(chunks[0])

  assert current_chunk is not None
  last_chunk = current_chunk
  if merge_silence:
    if last_chunk.is_silence:
      merge_last_chunk = last_chunk.size < min_duration
      if merge_last_chunk:
        previous_chunk = final_chunks[-1]
        previous_chunk.size += last_chunk.size
      else:
        final_chunks.append(last_chunk)
    else:
      final_chunks.append()

  process_last_chunk = (not merge_silence and not last_chunk.is_silence) or (
    merge_silence and last_chunk.is_silence)
  if process_last_chunk:
    assert len(final_chunks) >= 1
    merge_last_chunk = last_chunk.size < min_duration
    if merge_last_chunk:
      previous_chunk = final_chunks[-1]
      previous_chunk.size += last_chunk.size
    else:
      final_chunks.append(last_chunk)

  return final_chunks


def merge_same_coherent_chunks(chunks: List[Chunk]) -> List[Chunk]:
  if len(chunks) <= 1:
    return chunks

  current_samples: int = 0
  last_chunk: Optional[Chunk] = None
  splits: List[Chunk] = list()

  chunk: Chunk
  for chunk in tqdm(chunks):
    if last_chunk is None or chunk.is_silence == last_chunk.is_silence:
      current_samples += chunk.size
    else:
      c = Chunk(
        is_silence=last_chunk.is_silence,
        size=current_samples
      )
      splits.append(c)
      # splits.append((last_chunk, current_samples))
      current_samples = chunk.size
    last_chunk = chunk

  if current_samples > 0:
    c = Chunk(
      is_silence=last_chunk.is_silence,
      size=current_samples
    )
    splits.append(c)

  return splits


def chunk_wav(wav: np.ndarray, chunk_size: int) -> List[np.ndarray]:
  assert chunk_size > 0
  if chunk_size > len(wav):
    chunk_size = len(wav)
  its = len(wav) / chunk_size
  its = ceil(its)
  trim = 0
  res = []
  for _ in range(its):
    res.append(wav[trim:trim + chunk_size])
    trim += chunk_size
  return res


def get_dBFS_chunks(wav_chunks: List[np.ndarray], max_val: Union[int, float]) -> List[float]:
  dBFSs = [get_dBFS(x, max_val) for x in wav_chunks]
  return dBFSs


def mask_silence(wav: np.ndarray, silence_boundary: float, chunk_size: int):
  wav_chunks = chunk_wav(wav, chunk_size)
  max_value = -1 * get_min_value(wav.dtype)
  dBFSs = get_dBFS_chunks(wav_chunks, max_value)
  threshold = get_silence_threshold(dBFSs, silence_boundary)
  result = get_silence_chunks(wav_chunks, dBFSs, threshold)
  return result


def get_silence_threshold(dBFSs: List[float], silence_boundary: float) -> float:
  print(dBFSs)
  print(min(dBFSs))
  print(max(dBFSs))
  diff = abs(abs(min(dBFSs)) - abs(max(dBFSs)))
  threshold = diff * silence_boundary
  silence_threshold = min(dBFSs) + threshold
  return silence_threshold


def get_silence_chunks(wav_chunks: List[np.ndarray], dBFSs: List[float], silence_threshold: float):
  res: List[Chunk] = list()
  for chunk, dBFS in tqdm(zip(wav_chunks, dBFSs)):
    is_silence = dBFS < silence_threshold
    chunk_len = len(chunk)
    chunk = Chunk(
      size=chunk_len,
      is_silence=is_silence
    )
    res.append(chunk)

  return res
