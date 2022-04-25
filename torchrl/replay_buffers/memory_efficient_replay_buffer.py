import numpy as np
from .base import BaseReplayBuffer


class MemoryEfficientReplayBuffer(BaseReplayBuffer):
  """
  Use list to store LazyFrame object
  LazyFrame store reference of the numpy array returned by the env
  Avoid replicate store of the frames
  """

  def add_sample(self, sample_dict, **kwargs):
    for key in sample_dict:
      if not hasattr(self, "_" + key):
        self.__setattr__(
          "_" + key,
          [None for _ in range(self._max_replay_buffer_size)])
      self.__getattribute__("_" + key)[self._top] = sample_dict[key]
    self._advance()

  def encode_batchs(self, key, batch_indices):
    pointer = self.__getattribute__("_"+key)
    data = []
    for idx in batch_indices:
      data.append(pointer[idx])
    return np.array(data, dtype=np.float)

  def random_batch(self, batch_size, sample_key):
    indices = np.random.randint(0, self._size, batch_size)
    return_dict = {}
    for key in sample_key:
      return_dict[key] = self.encode_batchs(key, indices)

    return return_dict
